"""
jump.py  —  Water ski JUMP reframe to 9:16   (v2 — background-static model)
=============================================================================
KEY INSIGHT from reviewing raw footage: during the ramp/jump, the camera
operator holds the boat/camera essentially STILL — background (trees, ramp,
shoreline) does not move in frame. Only the skier moves, in a smooth
ballistic arc (up, across, down). This means we do NOT need aggressive
velocity-coasting during the air phase — we need a very gentle, heavily
dampened pan that barely moves, since chasing noisy YOLO detections on a
small distant airborne figure is what was causing the jumpy/snappy feel.

Behavior:
  - TRACKING: normal YOLO-driven tracking pre-ramp and post-landing
  - AIRBORNE: once skier height (small bbox, high in frame, low confidence
    motion) suggests they're airborne, switch to ultra-smooth slow pan
    using heavily averaged detections — never snaps, no velocity coasting
  - Detection averaging is much heavier during AIRBORNE to reject noise

Usage:
    python jump.py input.mov output.mp4

To restore audio (done automatically — see bottom of script):
    ffmpeg -i output.mp4 -i input.mov -c copy -map 0:v:0 -map 1:a:0 final.mp4
"""

import cv2
import argparse
import numpy as np
from collections import deque
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")

    # ── Detection ────────────────────────────────────────────────────────────
    p.add_argument("--conf", type=float, default=0.12,
                   help="""YOLO detection confidence threshold.
LOWER = detects skier more aggressively (more false positives from spray/boat).
HIGHER = only confident detections (may lose skier in spray/air). (default: 0.20)""")

    p.add_argument("--jump_limit", type=float, default=0.12,
                   help="""Max fraction of frame width a new detection can be from last known position.
LOWER = stricter, rejects more false detections.
HIGHER = looser, follows fast moves better but risks bad detections. (default: 0.30)""")

    # ── Smoothing (ground phase — before ramp / after landing) ────────────────
    p.add_argument("--smooth", type=float, default=0.18,
                   help="""EMA speed while skier is on the water (pre-ramp/post-landing).
LOWER = smoother/slower to follow. HIGHER = snappier. (default: 0.18)""")

    # ── Smoothing (air phase) ──────────────────────────────────────────────────
    p.add_argument("--air_smooth", type=float, default=0.045,
                   help="""EMA speed while skier is AIRBORNE. This is the key tuning knob
for the jump arc — the background is static during air time, so this should
be very slow/gentle. LOWER = barely pans (very smooth, may lag a fast arc).
HIGHER = pans more eagerly (smoother than ground smooth, but more reactive).
(default: 0.045)""")

    p.add_argument("--air_det_avg", type=int, default=10,
                   help="""How many recent airborne detections to average together before
moving the crop at all. Higher = rejects noisy single-frame detections
(very common for a small airborne figure), at the cost of a little lag.
(default: 10)""")

    p.add_argument("--max_speed", type=int, default=55,
                   help="""Hard cap on pixels the crop center can move per frame (full-res 4K px).
Applies at all times — this is what prevents any snap/jump outright.
LOWER = smoother, may lag fast motion. HIGHER = follows faster, risk of visible jump.
(default: 55)""")

    p.add_argument("--air_max_speed", type=int, default=25,
                   help="""Hard cap on pixels crop can move per frame WHILE AIRBORNE specifically.
Much lower than --max_speed since the background is static during air time —
there is no reason for the crop to move quickly. (default: 25)""")

    # ── Ground-phase lost-skier behavior ────────────────────────────────────────
    p.add_argument("--decay", type=float, default=0.92,
                   help="""How fast coasting velocity decays per frame when skier is lost
on the GROUND phase (pre-ramp / post-landing only — air phase doesn't coast).
LOWER = stops coasting sooner. HIGHER = coasts longer. (default: 0.78)""")

    p.add_argument("--coast_scale", type=float, default=0.75,
                   help="""Ground-phase coast strength when skier is lost.
0.0 = hold still. 1.0 = full velocity coast. (default: 0.4)""")

    p.add_argument("--miss_grace", type=int, default=4,
                   help="""Consecutive missed detections required before leaving TRACKING
and entering COASTING. A brief 1-3 frame flicker (spray, motion blur) is
common and should NOT trigger a full state-machine cycle — that rapid
TRACKING<->COASTING<->REACQUIRING flicker is what causes visible stutter.
LOWER = more reactive to real losses, but more prone to stutter on flicker.
HIGHER = smoother through brief flicker, slower to react to genuine loss.
(default: 4)""")

    p.add_argument("--min_reacquire", type=int, default=6,
                   help="""Frames to wait/confirm before trusting a new detection after
losing the skier on the ground phase. (default: 10)""")

    p.add_argument("--reacquire_frames", type=int, default=4,
                   help="Consecutive detections needed to confirm re-lock. (default: 4)")

    # ── Detection quality ────────────────────────────────────────────────────
    p.add_argument("--model", type=str, default="yolov8s.pt",
                   help="""YOLO model file. yolov8n.pt = fastest, least accurate.
yolov8s.pt = small model, notably better at small/distant/low-contrast
subjects, ~2-3x slower than nano. yolov8m.pt = medium, even more accurate,
much slower. For hazy/overcast footage with a small distant skier, yolov8s
or yolov8m catches detections that yolov8n misses entirely. (default: yolov8s.pt)""")

    p.add_argument("--detect_width", type=int, default=1280,
                   help="""Width (px) the frame is downscaled to before running YOLO.
Higher = better detection of small/distant subjects, slower processing.
960 was the old default (fast but misses small figures). 1280-1600 is
notably better for hazy/distant skier shots at the cost of speed.
(default: 1280)""")

    # ── Framing ─────────────────────────────────────────────────────────────
    p.add_argument("--headroom", type=float, default=0.45,
                   help="""Vertical anchor for skier in frame, fraction from top.
LOWER = more sky above (good for jump — gives room for the arc).
HIGHER = skier higher in frame. (default: 0.45)""")

    p.add_argument("--output_height", type=int, default=1920)

    return p.parse_args()


def out_dims(height):
    w = int(height * 9 / 16)
    return w + w % 2, height


def clamp_crop(cx, cy, cw, ch, fw, fh, headroom):
    x1 = int(cx - cw / 2)
    y1 = int(cy - ch * headroom)
    x1 = max(0, min(x1, fw - cw))
    y1 = max(0, min(y1, fh - ch))
    return x1, y1


def detect_skier_yolo(model, frame_small, conf, last_cx_norm, jump_limit_px, expected_box_h=None):
    """Returns (cx, cy, box_height) in frame_small coords, or None."""
    h, w = frame_small.shape[:2]
    top_mask    = int(h * 0.06)   # airborne skier can be high in frame — don't mask too much
    bottom_mask = int(h * 0.65)
    roi = frame_small[top_mask:bottom_mask, :]

    results = model(roi, classes=[0], verbose=False, conf=conf)
    boxes = []
    for r in results:
        for box in r.boxes:
            b = box.xyxy[0].cpu().numpy().copy()
            b[1] += top_mask
            b[3] += top_mask
            boxes.append(b)

    if not boxes:
        return None

    # Reject boxes whose center is in bottom 15% of ROI (boat/gunwale zone)
    y_cutoff = top_mask + (bottom_mask - top_mask) * 0.85
    boxes = [b for b in boxes if (b[1]+b[3])/2 < y_cutoff]
    if not boxes:
        return None

    # ── Size sanity filter ──────────────────────────────────────────────────
    # Reject any box whose height is wildly different from the recent
    # ground-phase average. A skier's apparent size changes gradually —
    # a box suddenly 3-5x normal size is almost always the boat, canopy,
    # or dock structure being misdetected as a person, NOT the skier.
    # This check is independent of jump_limit (position) because these
    # false positives can appear right where we'd expect the skier to be.
    if expected_box_h is not None and expected_box_h > 0:
        boxes = [b for b in boxes if (b[3]-b[1]) < expected_box_h * 2.2
                                   and (b[3]-b[1]) > expected_box_h * 0.25]
        if not boxes:
            return None

    if last_cx_norm is not None:
        last_cx_px = last_cx_norm * w
        filtered = [b for b in boxes
                    if abs((b[0]+b[2])/2 - last_cx_px) <= jump_limit_px]
        if filtered:
            boxes = filtered
        else:
            return None

    best = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
    cx = float((best[0]+best[2])/2)
    cy = float((best[1]+best[3])/2)
    box_h = float(best[3]-best[1])
    return cx, cy, box_h


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Cannot open {args.input}"); return

    FW    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FH    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS   = cap.get(cv2.CAP_PROP_FPS)
    TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    OUT_W, OUT_H = out_dims(args.output_height)
    scale  = min(FW / OUT_W, FH / OUT_H, 1.0)
    CROP_W = int(OUT_W * scale);  CROP_W -= CROP_W % 2
    CROP_H = int(OUT_H * scale);  CROP_H -= CROP_H % 2

    AW = args.detect_width
    AH = int(AW * FH / FW)
    AH -= AH % 2
    sx = FW / AW
    sy = FH / AH
    jump_limit_px = args.jump_limit * AW

    print(f"Input : {FW}x{FH} @ {FPS}fps ({TOTAL} frames, {TOTAL/FPS:.1f}s)")
    print(f"Crop  : {CROP_W}x{CROP_H}  Output: {OUT_W}x{OUT_H}")
    print(f"Detect: {AW}x{AH}  Model: {args.model}")
    print(f"Ground: smooth={args.smooth} decay={args.decay} coast_scale={args.coast_scale} max_speed={args.max_speed}")
    print(f"Air   : air_smooth={args.air_smooth} air_max_speed={args.air_max_speed} air_det_avg={args.air_det_avg}")
    print(f"Loading YOLO ({args.model})...")
    model = YOLO(args.model)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, FPS, (OUT_W, OUT_H))

    smooth_cx = FW / 2.0
    smooth_cy = FH * 0.42
    target_cx = smooth_cx
    target_cy = smooth_cy
    vel_x     = 0.0
    vel_y     = 0.0

    recent_dets   = deque(maxlen=8)
    last_cx_norm  = 0.5

    # Airborne detection is based on box height shrinking relative to a
    # rolling baseline — a smaller box at similar/higher Y position than
    # recent ground-phase boxes suggests the skier is airborne (farther
    # from camera vertically, or simply elevated and thus smaller/higher).
    ground_box_heights = deque(maxlen=20)
    air_det_buf_x = deque(maxlen=args.air_det_avg)
    air_det_buf_y = deque(maxlen=args.air_det_avg)

    state        = "TRACKING"   # TRACKING | COASTING | REACQUIRING | AIRBORNE
    lost_frames  = 0
    reacq_frames = 0
    pending_candidate = None
    air_lost_count = 0   # consecutive lost frames while airborne — used to exit AIRBORNE if lost too long
    miss_streak = 0       # consecutive missed detections in TRACKING — grace period before COASTING
    reacq_miss_streak = 0  # consecutive missed detections in REACQUIRING — grace period before COASTING

    yolo_hits   = 0
    coast_count = 0
    reacq_count = 0
    air_count   = 0

    print("Processing...\n")
    frame_num = 0

    # ── Debug logging — writes a CSV with one row per frame so we can
    # diagnose exactly what's happening at any timestamp without guessing.
    # Debug logging — uncomment to re-enable per-frame CSV diagnostics
    # debug_log = open(args.output + ".debug.csv", "w")
    # debug_log.write("frame,time_s,state,detected,box_h,is_airborne,target_cx,target_cy,smooth_cx,smooth_cy\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        if frame_num % 60 == 0:
            pct = frame_num / TOTAL * 100
            print(f"  {frame_num}/{TOTAL} ({pct:.0f}%)  [{state}]  yolo={yolo_hits} coast={coast_count} reacq={reacq_count} air={air_count}")

        small  = cv2.resize(frame, (AW, AH))
        expected_h = float(np.mean(ground_box_heights)) if len(ground_box_heights) >= 5 else None
        result = detect_skier_yolo(model, small, args.conf, last_cx_norm, jump_limit_px, expected_h)

        # ── Airborne heuristic ──────────────────────────────────────────────
        # A detection is "airborne-like" if its box height is meaningfully
        # smaller than the recent ground-phase average (skier appears
        # smaller when up and away) — OR if we're already in AIRBORNE state.
        is_airborne_detection = False
        if result is not None:
            _, _, box_h = result
            if len(ground_box_heights) >= 8:
                avg_ground_h = np.mean(ground_box_heights)
                # Much stricter threshold (was 0.72) — a hard cut toward the
                # ramp can shrink the box too, so require a MUCH bigger drop
                # before trusting this is actually airborne, not just a cut.
                if box_h < avg_ground_h * 0.45:
                    is_airborne_detection = True

        # ════════════════════════════════════════════════════════════════════
        if state == "TRACKING":
            if result is not None:
                rcx, rcy, box_h = result
                det_cx = rcx * sx
                det_cy = rcy * sy

                if is_airborne_detection:
                    # Transition into AIRBORNE mode
                    state = "AIRBORNE"
                    air_det_buf_x.clear()
                    air_det_buf_y.clear()
                    air_det_buf_x.append(det_cx)
                    air_det_buf_y.append(det_cy)
                    target_cx = det_cx
                    target_cy = det_cy
                    air_lost_count = 0
                    air_count += 1
                else:
                    ground_box_heights.append(box_h)
                    recent_dets.append((frame_num, det_cx, det_cy))
                    if len(recent_dets) >= 3:
                        ns = np.array([d[0] for d in recent_dets], dtype=float)
                        xs = np.array([d[1] for d in recent_dets], dtype=float)
                        ys = np.array([d[2] for d in recent_dets], dtype=float)
                        if ns[-1] - ns[0] > 0:
                            vel_x = float(np.polyfit(ns, xs, 1)[0])
                            vel_y = float(np.polyfit(ns, ys, 1)[0])
                    last_cx_norm = rcx / AW
                    target_cx = det_cx
                    target_cy = det_cy
                    lost_frames = 0
                    miss_streak = 0
                    yolo_hits += 1
            else:
                miss_streak += 1
                if miss_streak < args.miss_grace:
                    # Brief 1-2 frame flicker (spray/motion blur) — DON'T
                    # switch states yet. Just hold target where it was and
                    # keep using normal tracking smoothness. This is what
                    # kills the rapid TRACKING<->COASTING<->REACQUIRING
                    # flicker that caused stutter near landings.
                    pass
                else:
                    state = "COASTING"
                    lost_frames = 1
                    coast_count += 1
                    pending_candidate = None
                    target_cx += vel_x * args.coast_scale
                    target_cy += vel_y * args.coast_scale
                    vel_x *= args.decay
                    vel_y *= args.decay
                    target_cx = max(CROP_W/2, min(target_cx, FW - CROP_W/2))
                    target_cy = max(CROP_H/2, min(target_cy, FH - CROP_H/2))

        elif state == "AIRBORNE":
            air_count += 1
            if result is not None:
                rcx, rcy, box_h = result
                det_cx = rcx * sx
                det_cy = rcy * sy

                air_det_buf_x.append(det_cx)
                air_det_buf_y.append(det_cy)
                # Heavily averaged position — this is what kills the jumpiness
                target_cx = float(np.mean(air_det_buf_x))
                target_cy = float(np.mean(air_det_buf_y))
                last_cx_norm = rcx / AW
                air_lost_count = 0

                # Exit airborne once box height returns to ground-phase size
                if len(ground_box_heights) >= 5:
                    avg_ground_h = np.mean(ground_box_heights)
                    if box_h >= avg_ground_h * 0.85:
                        # Landed — back to normal tracking
                        state = "TRACKING"
                        miss_streak = 0
                        ground_box_heights.append(box_h)
                        recent_dets.clear()
                        recent_dets.append((frame_num, det_cx, det_cy))
                        vel_x = 0.0
                        vel_y = 0.0
            else:
                air_lost_count += 1
                # Briefly lost mid-air (occlusion) — just hold the averaged
                # position, do NOT coast. The background is static, the
                # skier's arc is smooth, holding is the safest bet.
                if air_lost_count > 25:
                    # Lost airborne skier for too long (>~0.8s) — fall back
                    # to ground coasting logic to avoid getting stuck.
                    state = "COASTING"
                    lost_frames = 1
                    coast_count += 1
                    pending_candidate = None

        elif state == "COASTING":
            coast_count += 1
            lost_frames += 1

            if result is not None:
                rcx, rcy, box_h = result
                cand_cx = rcx * sx
                cand_cy = rcy * sy

                # Sanity check: only trust this single-frame detection
                # immediately if it's reasonably close to where the coast
                # trajectory currently is. A detection that's wildly far
                # from the current trajectory is far more likely to be a
                # false positive (background structure, far-bank object)
                # than a real skier teleport — reject those outright rather
                # than accepting them as instant truth.
                dist_from_coast = np.hypot(cand_cx - target_cx, cand_cy - target_cy)
                max_plausible_jump = AW * sx * 0.18   # ~18% of frame width

                if dist_from_coast <= max_plausible_jump:
                    # Close enough to trust as a real correction
                    target_cx = cand_cx
                    target_cy = cand_cy
                    last_cx_norm = rcx / AW
                else:
                    # Too far to trust on a single frame — keep coasting
                    # normally, but remember this as a pending candidate
                    # for the existing 2-frame agreement check below.
                    target_cx += vel_x * args.coast_scale
                    target_cy += vel_y * args.coast_scale
                    vel_x *= args.decay
                    vel_y *= args.decay
                    target_cx = max(CROP_W/2, min(target_cx, FW - CROP_W/2))
                    target_cy = max(CROP_H/2, min(target_cy, FH - CROP_H/2))
            else:
                target_cx += vel_x * args.coast_scale
                target_cy += vel_y * args.coast_scale
                vel_x *= args.decay
                vel_y *= args.decay
                target_cx = max(CROP_W/2, min(target_cx, FW - CROP_W/2))
                target_cy = max(CROP_H/2, min(target_cy, FH - CROP_H/2))

            if result is not None and lost_frames >= args.min_reacquire:
                rcx, rcy, box_h = result
                cand_cx = rcx * sx
                cand_cy = rcy * sy
                if pending_candidate is not None:
                    prev_cx, prev_cy = pending_candidate
                    agree_dist = np.hypot(cand_cx - prev_cx, cand_cy - prev_cy)
                    if agree_dist < AW * sx * 0.06:
                        state = "REACQUIRING"
                        reacq_frames = 0
                        reacq_miss_streak = 0
                        target_cx = cand_cx
                        target_cy = cand_cy
                        last_cx_norm = rcx / AW
                        recent_dets.clear()
                        recent_dets.append((frame_num, cand_cx, cand_cy))
                        pending_candidate = None
                    else:
                        pending_candidate = (cand_cx, cand_cy)
                else:
                    pending_candidate = (cand_cx, cand_cy)
            else:
                pending_candidate = None

        elif state == "REACQUIRING":
            reacq_count += 1
            if result is not None:
                rcx, rcy, box_h = result
                det_cx = rcx * sx
                det_cy = rcy * sy
                target_cx = det_cx
                target_cy = det_cy
                last_cx_norm = rcx / AW
                recent_dets.append((frame_num, det_cx, det_cy))
                reacq_frames += 1
                reacq_miss_streak = 0
                if reacq_frames >= args.reacquire_frames:
                    if len(recent_dets) >= 3:
                        ns = np.array([d[0] for d in recent_dets], dtype=float)
                        xs = np.array([d[1] for d in recent_dets], dtype=float)
                        ys = np.array([d[2] for d in recent_dets], dtype=float)
                        if ns[-1] - ns[0] > 0:
                            vel_x = float(np.polyfit(ns, xs, 1)[0])
                            vel_y = float(np.polyfit(ns, ys, 1)[0])
                    ground_box_heights.append(box_h)
                    state = "TRACKING"
                    miss_streak = 0
            else:
                # Tolerate a brief 1-frame flicker mid-glide without bouncing
                # back to COASTING — that bounce was resetting reacq_frames
                # progress and causing extra visible catch-up motion on top
                # of the genuine detection gap.
                reacq_miss_streak += 1
                if reacq_miss_streak >= 2:
                    state = "COASTING"
                    lost_frames += reacq_frames
                    reacq_miss_streak = 0
                # else: hold target_cx/cy where they are, stay in REACQUIRING

        # ── EMA + speed clamp ────────────────────────────────────────────────
        if state == "AIRBORNE":
            ema = args.air_smooth
            cap_speed = args.air_max_speed
        elif state == "REACQUIRING":
            ema = 0.05
            cap_speed = args.max_speed
        else:
            ema = args.smooth
            cap_speed = args.max_speed

        new_cx = smooth_cx + ema * (target_cx - smooth_cx)
        new_cy = smooth_cy + ema * (target_cy - smooth_cy)

        move_x = new_cx - smooth_cx
        move_y = new_cy - smooth_cy
        dist = np.hypot(move_x, move_y)
        if dist > cap_speed:
            f = cap_speed / dist
            move_x *= f
            move_y *= f

        smooth_cx += move_x
        smooth_cy += move_y

        box_h_log = result[2] if result is not None else -1
        # debug_log.write(f"{frame_num},{frame_num/FPS:.3f},{state},{result is not None},{box_h_log:.1f},{is_airborne_detection},{target_cx:.1f},{target_cy:.1f},{smooth_cx:.1f},{smooth_cy:.1f}\n")

        x1, y1 = clamp_crop(smooth_cx, smooth_cy, CROP_W, CROP_H, FW, FH, args.headroom)
        cropped = frame[y1:y1+CROP_H, x1:x1+CROP_W]
        resized = cv2.resize(cropped, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
        writer.write(resized)

    cap.release()
    writer.release()
    # debug_log.close()

    print(f"\nDone -> {args.output}")
    print(f"Tracking: {yolo_hits} | Coasting: {coast_count} | Reacquiring: {reacq_count} | Airborne: {air_count}")

    # ── Auto-merge audio ─────────────────────────────────────────────────────
    import subprocess, os
    base, ext = os.path.splitext(args.output)
    final_output = base + "_audio" + ext
    print(f"\nMerging original audio into: {final_output} ...")
    cmd = ["ffmpeg", "-y", "-i", args.output, "-i", args.input,
           "-c", "copy", "-map", "0:v:0", "-map", "1:a:0", final_output]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        print(f"Success! Final video with audio: {final_output}")
        os.remove(args.output)
    else:
        print("ffmpeg not found or failed. Install ffmpeg and run manually:")
        print(f"  ffmpeg -i {args.output} -i {args.input} -c copy -map 0:v:0 -map 1:a:0 {final_output}")


if __name__ == "__main__":
    main()
