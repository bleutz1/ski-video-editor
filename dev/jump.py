"""
jump.py  —  Water ski JUMP reframe to 9:16  (Kalman filter edition)
====================================================================
KEY INSIGHT: during the ramp/jump the camera holds essentially still —
background does not move. Only the skier moves in a smooth ballistic arc.

Kalman replaces EMA throughout. The filter's process noise is dialled down
hard during AIRBORNE (ballistic arc = very predictable motion model), giving
the same ultra-smooth air-phase behaviour as the old air_smooth parameter
but derived from physics rather than an arbitrary EMA constant.

States: TRACKING → COASTING → REACQUIRING | TRACKING → AIRBORNE → TRACKING

Usage:
    python jump.py input.mov output.mp4

Audio is merged automatically via ffmpeg at the end.
"""

import cv2
import argparse
import numpy as np
from collections import deque
from ultralytics import YOLO


# ── Kalman filter ────────────────────────────────────────────────────────────
class KalmanTracker:
    """
    2D Kalman filter tracking crop-center position (cx, cy).

    State vector:  [cx, cy, vx, vy]
    Measurement:   [cx, cy]   (from YOLO detection)

    process_noise     — trust in the motion model (higher = adapts faster).
    measurement_noise — trust in each YOLO detection (higher = smoother/laggier).

    Hot-swap noise params via set_noise() to change behaviour per state:
      - AIRBORNE: very low process_noise (ballistic arc is predictable)
      - REACQUIRING: high measurement_noise (ease onto skier, don't snap)
    """

    def __init__(self, cx, cy, process_noise=2.0, measurement_noise=30.0):
        self.x = np.array([cx, cy, 0.0, 0.0], dtype=float)

        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        q = process_noise
        self.Q = np.diag([q, q, q * 0.5, q * 0.5])

        r = measurement_noise
        self.R = np.diag([r, r])

        self.P = np.eye(4) * 500.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0], self.x[1]

    def update(self, cx_meas, cy_meas):
        z = np.array([cx_meas, cy_meas])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[0], self.x[1]

    @property
    def position(self):
        return self.x[0], self.x[1]

    @property
    def velocity(self):
        return self.x[2], self.x[3]

    def set_noise(self, process_noise=None, measurement_noise=None):
        if process_noise is not None:
            q = process_noise
            self.Q = np.diag([q, q, q * 0.5, q * 0.5])
        if measurement_noise is not None:
            r = measurement_noise
            self.R = np.diag([r, r])


# ── Argument parsing ─────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")

    # ── Detection ────────────────────────────────────────────────────────────
    p.add_argument("--conf", type=float, default=0.12,
                   help="YOLO confidence threshold. Lower = more detections, more noise. (default: 0.12)")

    p.add_argument("--jump_limit", type=float, default=0.12,
                   help="Max fraction of frame width a detection can be from last known position. (default: 0.12)")

    # ── Kalman noise — ground phase ───────────────────────────────────────────
    p.add_argument("--process_noise", type=float, default=2.0,
                   help="""Kalman process noise (ground phase).
LOWER = smoother coast, slower to change direction.
HIGHER = adapts faster to direction changes. (default: 2.0)""")

    p.add_argument("--measurement_noise", type=float, default=30.0,
                   help="""Kalman measurement noise (ground phase).
LOWER = snappier, follows detections closely.
HIGHER = smoother, lags more. (default: 30.0)""")

    # ── Kalman noise — air phase ──────────────────────────────────────────────
    p.add_argument("--air_process_noise", type=float, default=0.3,
                   help="""Kalman process noise during AIRBORNE.
Very low = filter trusts its own ballistic arc prediction, barely reacts to
noisy detections of the small distant skier. Raise if arc tracking lags badly.
(default: 0.3)""")

    p.add_argument("--air_measurement_noise", type=float, default=120.0,
                   help="""Kalman measurement noise during AIRBORNE.
Very high = individual YOLO hits have minimal influence during the arc.
The filter mostly flies on its own velocity estimate. (default: 120.0)""")

    p.add_argument("--air_det_avg", type=int, default=10,
                   help="Airborne detection averaging window before Kalman update. (default: 10)")

    # ── Reacquire noise ───────────────────────────────────────────────────────
    p.add_argument("--reacquire_noise", type=float, default=80.0,
                   help="Measurement noise during REACQUIRING — eases onto skier, prevents snap. (default: 80.0)")

    # ── Speed clamps ─────────────────────────────────────────────────────────
    p.add_argument("--max_speed", type=int, default=55,
                   help="Hard per-frame pixel cap (ground phase). (default: 55)")

    p.add_argument("--air_max_speed", type=int, default=25,
                   help="Hard per-frame pixel cap while AIRBORNE — background is static, no reason to move fast. (default: 25)")

    # ── Lost-skier behavior ──────────────────────────────────────────────────
    p.add_argument("--miss_grace", type=int, default=4,
                   help="Consecutive missed frames before leaving TRACKING → COASTING. (default: 4)")

    p.add_argument("--min_reacquire", type=int, default=6,
                   help="Minimum coast frames before trusting a new detection. (default: 6)")

    p.add_argument("--reacquire_frames", type=int, default=4,
                   help="Consecutive detections needed to confirm re-lock. (default: 4)")

    # ── Model / detection ────────────────────────────────────────────────────
    p.add_argument("--model", type=str, default="yolov8s.pt")
    p.add_argument("--detect_width", type=int, default=1280)

    # ── Framing ─────────────────────────────────────────────────────────────
    p.add_argument("--headroom", type=float, default=0.45,
                   help="Vertical anchor for skier (fraction from top). Lower = more sky for arc. (default: 0.45)")

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
    top_mask    = int(h * 0.06)   # low mask — airborne skier can be high in frame
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

    y_cutoff = top_mask + (bottom_mask - top_mask) * 0.85
    boxes = [b for b in boxes if (b[1]+b[3])/2 < y_cutoff]
    if not boxes:
        return None

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
    AH = int(AW * FH / FW); AH -= AH % 2
    sx = FW / AW
    sy = FH / AH
    jump_limit_px = args.jump_limit * AW

    print(f"Input : {FW}x{FH} @ {FPS}fps ({TOTAL} frames, {TOTAL/FPS:.1f}s)")
    print(f"Crop  : {CROP_W}x{CROP_H}  Output: {OUT_W}x{OUT_H}")
    print(f"Detect: {AW}x{AH}  Model: {args.model}")
    print(f"Ground: process_noise={args.process_noise}  measurement_noise={args.measurement_noise}  max_speed={args.max_speed}")
    print(f"Air   : air_process_noise={args.air_process_noise}  air_measurement_noise={args.air_measurement_noise}  air_max_speed={args.air_max_speed}")
    print(f"\nLoading YOLO ({args.model})...")
    model = YOLO(args.model)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, FPS, (OUT_W, OUT_H))

    # ── Kalman filter — initialised at frame center ──────────────────────────
    kf = KalmanTracker(
        cx=FW / 2.0,
        cy=FH * 0.42,
        process_noise=args.process_noise,
        measurement_noise=args.measurement_noise,
    )

    smooth_cx, smooth_cy = kf.position
    last_cx_norm = 0.5

    # Airborne detection averaging (separate from ground det_buf)
    air_det_buf_x = deque(maxlen=args.air_det_avg)
    air_det_buf_y = deque(maxlen=args.air_det_avg)

    # Ground detection averaging buffer
    det_avg_window = 5
    det_buf_x = deque(maxlen=det_avg_window)
    det_buf_y = deque(maxlen=det_avg_window)

    recent_dets        = deque(maxlen=8)
    ground_box_heights = deque(maxlen=20)

    state             = "TRACKING"   # TRACKING | COASTING | REACQUIRING | AIRBORNE
    lost_frames       = 0
    reacq_frames      = 0
    air_lost_count    = 0
    miss_streak       = 0
    reacq_miss_streak = 0
    pending_candidate = None

    yolo_hits   = 0
    coast_count = 0
    reacq_count = 0
    air_count   = 0

    print("Processing...\n")
    frame_num = 0

    # debug_log = open(args.output + ".debug.csv", "w")
    # debug_log.write("frame,time_s,state,detected,box_h,is_airborne,target_cx,target_cy,smooth_cx,smooth_cy,kf_vx,kf_vy\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        if frame_num % 60 == 0:
            pct = frame_num / TOTAL * 100
            vx, vy = kf.velocity
            print(f"  {frame_num}/{TOTAL} ({pct:.0f}%)  [{state}]  "
                  f"yolo={yolo_hits} coast={coast_count} reacq={reacq_count} air={air_count}  "
                  f"kf_vel=({vx:.1f},{vy:.1f})")

        small      = cv2.resize(frame, (AW, AH))
        expected_h = float(np.mean(ground_box_heights)) if len(ground_box_heights) >= 5 else None
        result     = detect_skier_yolo(model, small, args.conf, last_cx_norm, jump_limit_px, expected_h)

        # ── Airborne heuristic ───────────────────────────────────────────────
        is_airborne_detection = False
        if result is not None:
            _, _, box_h = result
            if len(ground_box_heights) >= 8:
                avg_ground_h = np.mean(ground_box_heights)
                if box_h < avg_ground_h * 0.45:
                    is_airborne_detection = True

        # ── Always predict first ─────────────────────────────────────────────
        kf.predict()

        # ════════════════════════════════════════════════════════════════════
        if state == "TRACKING":
            if result is not None:
                rcx, rcy, box_h = result
                det_cx = rcx * sx
                det_cy = rcy * sy

                if is_airborne_detection:
                    # ── Enter AIRBORNE ────────────────────────────────────────
                    state = "AIRBORNE"
                    air_det_buf_x.clear()
                    air_det_buf_y.clear()
                    air_det_buf_x.append(det_cx)
                    air_det_buf_y.append(det_cy)
                    # Tighten process noise (trust the ballistic arc),
                    # raise measurement noise (ignore noisy detections)
                    kf.set_noise(
                        process_noise=args.air_process_noise,
                        measurement_noise=args.air_measurement_noise,
                    )
                    kf.update(det_cx, det_cy)
                    last_cx_norm   = rcx / AW
                    air_lost_count = 0
                    air_count     += 1
                else:
                    det_buf_x.append(det_cx)
                    det_buf_y.append(det_cy)
                    avg_cx = float(np.mean(det_buf_x))
                    avg_cy = float(np.mean(det_buf_y))
                    kf.update(avg_cx, avg_cy)
                    ground_box_heights.append(box_h)
                    recent_dets.append((frame_num, det_cx, det_cy))
                    last_cx_norm = rcx / AW
                    lost_frames  = 0
                    miss_streak  = 0
                    yolo_hits   += 1
            else:
                miss_streak += 1
                if miss_streak >= args.miss_grace:
                    state        = "COASTING"
                    lost_frames  = 1
                    coast_count += 1
                    pending_candidate = None

        elif state == "AIRBORNE":
            air_count += 1
            if result is not None:
                rcx, rcy, box_h = result
                det_cx = rcx * sx
                det_cy = rcy * sy

                air_det_buf_x.append(det_cx)
                air_det_buf_y.append(det_cy)
                avg_cx = float(np.mean(air_det_buf_x))
                avg_cy = float(np.mean(air_det_buf_y))

                # Update with heavily averaged position — noisy single frames
                # have very little influence thanks to air_measurement_noise
                kf.update(avg_cx, avg_cy)
                last_cx_norm   = rcx / AW
                air_lost_count = 0

                # Exit AIRBORNE once box height returns to near ground size
                if len(ground_box_heights) >= 5:
                    avg_ground_h = np.mean(ground_box_heights)
                    if box_h >= avg_ground_h * 0.85:
                        # Landed — restore ground-phase noise
                        kf.set_noise(
                            process_noise=args.process_noise,
                            measurement_noise=args.measurement_noise,
                        )
                        state       = "TRACKING"
                        miss_streak = 0
                        ground_box_heights.append(box_h)
                        recent_dets.clear()
                        recent_dets.append((frame_num, det_cx, det_cy))
                        # Zero velocity on landing — ground phase starts fresh
                        kf.x[2] = 0.0
                        kf.x[3] = 0.0
            else:
                air_lost_count += 1
                # Hold — Kalman predict (already called) advances on arc
                if air_lost_count > 25:
                    # Lost for >~0.8s airborne — fall back to ground coasting
                    kf.set_noise(
                        process_noise=args.process_noise,
                        measurement_noise=args.measurement_noise,
                    )
                    state       = "COASTING"
                    lost_frames = 1
                    coast_count += 1
                    pending_candidate = None

        elif state == "COASTING":
            coast_count += 1
            lost_frames += 1
            # Kalman predict already called — velocity-based coast is free

            if result is not None:
                rcx, rcy, box_h = result
                cand_cx = rcx * sx
                cand_cy = rcy * sy

                pred_cx, pred_cy = kf.position
                dist_from_pred   = np.hypot(cand_cx - pred_cx, cand_cy - pred_cy)
                max_plausible    = AW * sx * 0.18

                if dist_from_pred <= max_plausible:
                    kf.update(cand_cx, cand_cy)
                    last_cx_norm = rcx / AW

                # Transition to REACQUIRING after enough coast frames
                if lost_frames >= args.min_reacquire:
                    if pending_candidate is not None:
                        prev_cx, prev_cy = pending_candidate
                        agree_dist = np.hypot(cand_cx - prev_cx, cand_cy - prev_cy)
                        if agree_dist < AW * sx * 0.06:
                            recent_dets.clear()
                            det_buf_x.clear()
                            det_buf_y.clear()
                            recent_dets.append((frame_num, cand_cx, cand_cy))
                            state        = "REACQUIRING"
                            reacq_frames = 0
                            reacq_miss_streak = 0
                            last_cx_norm = rcx / AW
                            pending_candidate = None
                            kf.set_noise(measurement_noise=args.reacquire_noise)
                        else:
                            pending_candidate = (cand_cx, cand_cy)
                    else:
                        pending_candidate = (cand_cx, cand_cy)
                else:
                    pending_candidate = None
            else:
                pending_candidate = None

        elif state == "REACQUIRING":
            reacq_count += 1
            if result is not None:
                rcx, rcy, box_h = result
                det_cx = rcx * sx
                det_cy = rcy * sy

                kf.update(det_cx, det_cy)
                last_cx_norm = rcx / AW
                recent_dets.append((frame_num, det_cx, det_cy))
                reacq_frames      += 1
                reacq_miss_streak  = 0

                if reacq_frames >= args.reacquire_frames:
                    ground_box_heights.append(box_h)
                    kf.set_noise(measurement_noise=args.measurement_noise)
                    state       = "TRACKING"
                    miss_streak = 0
            else:
                reacq_miss_streak += 1
                if reacq_miss_streak >= 2:
                    state        = "COASTING"
                    lost_frames += reacq_frames
                    reacq_miss_streak = 0
                    kf.set_noise(measurement_noise=args.measurement_noise)

        # ── Read Kalman position ─────────────────────────────────────────────
        new_cx, new_cy = kf.position

        # ── HARD SPEED CLAMP — always applied, nothing bypasses this ─────────
        cap_speed = args.air_max_speed if state == "AIRBORNE" else args.max_speed
        move_x = new_cx - smooth_cx
        move_y = new_cy - smooth_cy
        dist   = np.hypot(move_x, move_y)
        if dist > cap_speed:
            f       = cap_speed / dist
            move_x *= f
            move_y *= f

        smooth_cx += move_x
        smooth_cy += move_y

        # Sync Kalman to clamped position so it doesn't drift from reality
        kf.x[0] = smooth_cx
        kf.x[1] = smooth_cy

        box_h_log = result[2] if result is not None else -1
        vx, vy = kf.velocity
        # debug_log.write(f"{frame_num},{frame_num/FPS:.3f},{state},{result is not None},{box_h_log:.1f},{is_airborne_detection},{new_cx:.1f},{new_cy:.1f},{smooth_cx:.1f},{smooth_cy:.1f},{vx:.2f},{vy:.2f}\n")

        x1, y1 = clamp_crop(smooth_cx, smooth_cy, CROP_W, CROP_H, FW, FH, args.headroom)
        cropped = frame[y1:y1+CROP_H, x1:x1+CROP_W]
        resized = cv2.resize(cropped, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
        writer.write(resized)

    cap.release()
    writer.release()
    # debug_log.close()

    print(f"\nDone → {args.output}")
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
