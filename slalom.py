"""
slalom.py  —  Slalom water ski reframe to 9:16
=====================================================
Usage:
    python auto_reframe.py input.mov output.mp4 --mode slalom
    python auto_reframe.py input.mov output.mp4 --mode jump

Modes:
    slalom  Fast side-to-side. Coasts in direction skier was moving when lost.
    jump    Skier goes up/down roughly center. Holds position when lost.

To restore audio after:
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
    p.add_argument("--conf", type=float, default=0.25,
                   help="""YOLO detection confidence threshold.
LOWER (e.g. 0.10) = detects skier more aggressively, may cause false positives from spray/boat.
HIGHER (e.g. 0.25) = only confident detections, may lose skier in spray. (default: 0.15)""")

    p.add_argument("--jump_limit", type=float, default=0.12,
                   help="""Max fraction of frame width a new detection can be from last known position.
LOWER (e.g. 0.15) = stricter, rejects more false detections, may miss fast swings.
HIGHER (e.g. 0.40) = looser, follows fast moves better, may accept bad detections. (default: 0.25)""")

    # ── Smoothing ────────────────────────────────────────────────────────────
    p.add_argument("--smooth", type=float, default=0.15,
                   help="""EMA speed during active tracking (how snappily crop follows skier).
LOWER (e.g. 0.10) = smoother/slower to follow, skier may drift to edge on fast moves.
HIGHER (e.g. 0.35) = snappier/faster follow, may feel jittery. (default: 0.22)""")

    p.add_argument("--reacquire_smooth", type=float, default=0.03,
                   help="""EMA speed when gliding back to skier after losing them.
LOWER (e.g. 0.03) = very slow gentle return, takes longer to center on skier.
HIGHER (e.g. 0.12) = faster return, may still feel like a jump. (default: 0.06)""")

    p.add_argument("--max_speed", type=int, default=120,
                   help="""Hard cap on pixels the crop center can move per frame (full-res 4K pixels).
This prevents ALL jumping regardless of detection quality.
LOWER (e.g. 50) = smoother but may not keep up with very fast skier.
HIGHER (e.g. 120) = follows faster, may allow small jumps. (default: 80)""")

    # ── Lost-skier behavior ──────────────────────────────────────────────────
    p.add_argument("--decay", type=float, default=0.82,
                   help="""How fast the coasting velocity slows down per frame when skier is lost.
LOWER (e.g. 0.70) = velocity dies quickly, frame holds sooner. Good for jump.
HIGHER (e.g. 0.95) = coasts much longer before stopping. Good for fast slalom. (default: 0.82)""")

    p.add_argument("--coast_scale", type=float, default=0.65,
                   help="""How much of the last velocity to apply per frame when coasting.
Overrides the mode default if set.
0.0 = hold completely still when lost (best for jump).
1.0 = full velocity coast (best for fast slalom).
0.4 = balanced middle ground. (mode defaults: slalom=0.55, jump=0.0)""")

    p.add_argument("--min_reacquire", type=int, default=12,
                   help="""Frames to ignore new detections after losing skier.
Prevents immediately jumping to a new detection (which may be wrong).
LOWER (e.g. 4) = re-locks onto skier sooner, may jump if detection was wrong.
HIGHER (e.g. 15) = coasts longer before re-locking, smoother but slower to recover.
(mode defaults: slalom=8, jump=12)""")

    p.add_argument("--reacquire_frames", type=int, default=5,
                   help="""Consecutive detections needed before switching back to full tracking.
LOWER (e.g. 2) = locks back on quickly, may jump if detections briefly flicker.
HIGHER (e.g. 8) = waits for more evidence, slower to snap back. (default: 5)""")

    p.add_argument("--miss_grace", type=int, default=4,
                   help="""Consecutive missed detections required before leaving TRACKING
and entering COASTING. A brief 1-3 frame flicker is common and shouldn't
trigger a full state-machine cycle. (default: 4)""")

    # ── Framing ─────────────────────────────────────────────────────────────
    p.add_argument("--headroom", type=float, default=0.48,
                   help="""Vertical position of skier in frame (as fraction of crop height from top).
LOWER (e.g. 0.38) = more sky above skier, skier lower in frame.
HIGHER (e.g. 0.55) = skier higher in frame, more water below. (default: 0.48)""")

    p.add_argument("--output_height", type=int, default=1920,
                   help="Output height in pixels. Width auto-set to 9:16. (default: 1920 → 1080x1920)")

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
    h, w = frame_small.shape[:2]
    # top_mask: exclude sky/canopy at top. Raise (e.g. 0.15) if canopy detected as person.
    # bottom_mask: exclude boat/gunwale. LOWER this value (e.g. 0.62) if boat still sneaks in.
    top_mask    = int(h * 0.12)
    bottom_mask = int(h * 0.65)   # tighter than before — keeps boat gunwale out
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

    # Reject any box whose CENTER Y is in the bottom 15% of the ROI —
    # that zone is water/boat edge, not the skier's body.
    roi_bottom_y = bottom_mask
    y_cutoff = top_mask + (roi_bottom_y - top_mask) * 0.85
    boxes = [b for b in boxes if (b[1]+b[3])/2 < y_cutoff]
    if not boxes:
        return None

    # ── Size sanity filter ──────────────────────────────────────────────────
    # Reject any box whose height is wildly different from the recent
    # ground-phase average. A skier's apparent size changes gradually —
    # a box suddenly 3-5x normal size is almost always the boat, canopy,
    # or dock structure being misdetected as a person, NOT the skier.
    if expected_box_h is not None and expected_box_h > 0:
        boxes = [b for b in boxes if (b[3]-b[1]) < expected_box_h * 2.2
                                   and (b[3]-b[1]) > expected_box_h * 0.25]
        if not boxes:
            return None

    if last_cx_norm is not None:
        last_cx_px = last_cx_norm * w
        filtered = [b for b in boxes
                    if abs((b[0]+b[2])/2 - last_cx_px) <= jump_limit_px]
        if not filtered:
            return None
        boxes = filtered

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

    AW, AH = 1280, 720
    sx = FW / AW
    sy = FH / AH
    jump_limit_px = args.jump_limit * AW

    print(f"Input : {FW}x{FH} @ {FPS}fps ({TOTAL} frames, {TOTAL/FPS:.1f}s)")
    print(f"Crop  : {CROP_W}x{CROP_H}  Output: {OUT_W}x{OUT_H}")
    print(f"conf={args.conf}  jump_limit={args.jump_limit}  max_speed={args.max_speed}px/f")
    print(f"smooth={args.smooth}  reacquire_smooth={args.reacquire_smooth}")
    print(f"decay={args.decay}  coast_scale={args.coast_scale}  min_reacquire={args.min_reacquire}")
    print("\nLoading YOLO (yolov8s.pt)...")
    model = YOLO("yolov8s.pt")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, FPS, (OUT_W, OUT_H))

    smooth_cx = FW / 2.0
    smooth_cy = FH * 0.45
    target_cx = smooth_cx
    target_cy = smooth_cy
    vel_x     = 0.0
    vel_y     = 0.0

    recent_dets  = deque(maxlen=8)
    last_cx_norm = 0.5

    # Detection averaging buffer — smooths YOLO's frame-to-frame jitter
    # before it reaches the EMA. This is what eliminates stutter.
    # Increase det_avg_window (e.g. 7-9) for smoother but slightly laggier tracking.
    # Decrease (e.g. 3) for more responsive but potentially stuttery tracking.
    det_avg_window = 5
    det_buf_x = deque(maxlen=det_avg_window)
    det_buf_y = deque(maxlen=det_avg_window)

    state        = "TRACKING"
    lost_frames  = 0
    reacq_frames = 0
    miss_streak  = 0   # consecutive missed detections in TRACKING — grace before COASTING
    reacq_miss_streak = 0  # consecutive missed detections in REACQUIRING — grace before COASTING
    ground_box_heights = deque(maxlen=20)

    yolo_hits   = 0
    coast_count = 0
    reacq_count = 0

    print("Processing...\n")
    frame_num = 0

    # Debug logging — matches jump.py's format for direct comparison
    # Debug logging — uncomment to re-enable per-frame CSV diagnostics
    debug_log = open(args.output + ".debug.csv", "w")
    debug_log.write("frame,time_s,state,detected,box_h,target_cx,target_cy,smooth_cx,smooth_cy\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        if frame_num % 60 == 0:
            pct   = frame_num / TOTAL * 100
            speed = np.hypot(vel_x, vel_y)
            print(f"  {frame_num}/{TOTAL} ({pct:.0f}%)  [{state}]  yolo={yolo_hits}  coast={coast_count}  reacq={reacq_count}  spd={speed:.1f}px/f")

        small  = cv2.resize(frame, (AW, AH))
        expected_h = float(np.mean(ground_box_heights)) if len(ground_box_heights) >= 5 else None
        result = detect_skier_yolo(model, small, args.conf, last_cx_norm, jump_limit_px, expected_h)

        if state == "TRACKING":
            if result is not None:
                rcx, rcy, box_h = result
                raw_cx = rcx * sx
                raw_cy = rcy * sy

                # Push raw detection into averaging buffer
                det_buf_x.append(raw_cx)
                det_buf_y.append(raw_cy)

                # Use averaged position as the actual detection — kills stutter
                det_cx = float(np.mean(det_buf_x))
                det_cy = float(np.mean(det_buf_y))

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
                target_cx    = det_cx
                target_cy    = det_cy
                lost_frames  = 0
                miss_streak  = 0
                yolo_hits   += 1
            else:
                miss_streak += 1
                if miss_streak < args.miss_grace:
                    # Brief 1-2 frame flicker — don't switch states yet,
                    # just hold target where it was.
                    pass
                else:
                    state        = "COASTING"
                    lost_frames  = 1
                    coast_count += 1
                    target_cx   += vel_x * args.coast_scale
                    target_cy   += vel_y * args.coast_scale
                    vel_x       *= args.decay
                    vel_y       *= args.decay
                    target_cx    = max(CROP_W/2, min(target_cx, FW - CROP_W/2))
                    target_cy    = max(CROP_H/2, min(target_cy, FH - CROP_H/2))

        elif state == "COASTING":
            coast_count += 1
            lost_frames += 1

            if result is not None:
                rcx, rcy, box_h = result
                cand_cx = rcx * sx
                cand_cy = rcy * sy

                # Sanity check: only trust this single-frame detection
                # immediately if it's reasonably close to the current coast
                # trajectory. A detection that's wildly far away is more
                # likely a false positive than a real teleport.
                dist_from_coast = np.hypot(cand_cx - target_cx, cand_cy - target_cy)
                max_plausible_jump = AW * sx * 0.18

                if dist_from_coast <= max_plausible_jump:
                    target_cx = cand_cx
                    target_cy = cand_cy
                    last_cx_norm = rcx / AW
                else:
                    target_cx += vel_x * args.coast_scale
                    target_cy += vel_y * args.coast_scale
                    vel_x *= args.decay
                    vel_y *= args.decay
                    target_cx = max(CROP_W/2, min(target_cx, FW - CROP_W/2))
                    target_cy = max(CROP_H/2, min(target_cy, FH - CROP_H/2))
            else:
                target_cx   += vel_x * args.coast_scale
                target_cy   += vel_y * args.coast_scale
                vel_x       *= args.decay
                vel_y       *= args.decay
                target_cx    = max(CROP_W/2, min(target_cx, FW - CROP_W/2))
                target_cy    = max(CROP_H/2, min(target_cy, FH - CROP_H/2))

            if result is not None and lost_frames >= args.min_reacquire:
                rcx, rcy, box_h = result
                det_cx       = rcx * sx
                det_cy       = rcy * sy
                state        = "REACQUIRING"
                reacq_frames = 0
                reacq_miss_streak = 0
                target_cx    = det_cx
                target_cy    = det_cy
                last_cx_norm = rcx / AW
                recent_dets.clear()
                det_buf_x.clear()
                det_buf_y.clear()
                recent_dets.append((frame_num, det_cx, det_cy))

        elif state == "REACQUIRING":
            reacq_count += 1
            if result is not None:
                rcx, rcy, box_h = result
                det_cx       = rcx * sx
                det_cy       = rcy * sy
                target_cx    = det_cx
                target_cy    = det_cy
                last_cx_norm = rcx / AW
                ground_box_heights.append(box_h)
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
                    state = "TRACKING"
                    miss_streak = 0
            else:
                # Tolerate a brief 1-frame flicker mid-glide without bouncing
                # back to COASTING — that bounce was resetting reacq_frames
                # progress and causing extra visible catch-up motion.
                reacq_miss_streak += 1
                if reacq_miss_streak >= 2:
                    state        = "COASTING"
                    lost_frames += reacq_frames
                    reacq_miss_streak = 0
                # else: hold target where it is, stay in REACQUIRING

        # EMA — slow glide during reacquire, snappy during tracking
        ema    = args.reacquire_smooth if state == "REACQUIRING" else args.smooth
        new_cx = smooth_cx + ema * (target_cx - smooth_cx)
        new_cy = smooth_cy + ema * (target_cy - smooth_cy)

        # HARD SPEED CLAMP — nothing gets through this, ever
        move_x = new_cx - smooth_cx
        move_y = new_cy - smooth_cy
        dist   = np.hypot(move_x, move_y)
        if dist > args.max_speed:
            f      = args.max_speed / dist
            move_x *= f
            move_y *= f

        smooth_cx += move_x
        smooth_cy += move_y

        box_h_log = result[2] if result is not None else -1
        debug_log.write(f"{frame_num},{frame_num/FPS:.3f},{state},{result is not None},{box_h_log:.1f},{target_cx:.1f},{target_cy:.1f},{smooth_cx:.1f},{smooth_cy:.1f}\n")

        x1, y1 = clamp_crop(smooth_cx, smooth_cy, CROP_W, CROP_H, FW, FH, args.headroom)
        cropped = frame[y1:y1+CROP_H, x1:x1+CROP_W]
        resized = cv2.resize(cropped, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
        writer.write(resized)

    cap.release()
    writer.release()
    debug_log.close()

    print(f"\nDone \u2192 {args.output}")
    print(f"Tracking: {yolo_hits} | Coasting: {coast_count} | Reacquiring: {reacq_count}")

    # Auto-merge original audio using ffmpeg
    import subprocess, os
    base, ext = os.path.splitext(args.output)
    final_output = base + "_audio" + ext

    print(f"\nMerging original audio into: {final_output} ...")
    cmd = [
        "ffmpeg", "-y",
        "-i", args.output,
        "-i", args.input,
        "-c", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        final_output
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        print(f"Success! Final video with audio: {final_output}")
        os.remove(args.output)
    else:
        print("ffmpeg not found or failed. Install ffmpeg and run manually:")
        print(f"  ffmpeg -i {args.output} -i {args.input} -c copy -map 0:v:0 -map 1:a:0 {final_output}")


if __name__ == "__main__":
    main()
