"""
slalom.py  —  Slalom water ski reframe to 9:16
=====================================================
Usage:
    python slalom.py input.mov output.mp4

Tracking uses a Kalman filter (position + velocity state) instead of EMA.
During COASTING the filter predicts forward using its own velocity estimate —
no manual vel_x/vel_y decay needed. Corrections from YOLO detections are
weighted by measurement noise; the hard max_speed clamp remains as a final
safety net.

To restore audio (done automatically — see bottom of script):
    ffmpeg -i output.mp4 -i input.mov -c copy -map 0:v:0 -map 1:a:0 final.mp4
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

    process_noise  — how much we trust the motion model each frame.
                     Higher = filter adapts faster to real position changes,
                     but is also more sensitive to noisy detections.
                     Lower  = smoother, slower to change direction.

    measurement_noise — how much we trust a single YOLO detection.
                        Higher = detections have less influence (smoother
                        but lags more). Lower = snappier but jitterier.

    The filter always predicts forward every frame (even during COASTING),
    which is the key advantage over EMA — it uses its own velocity estimate
    rather than needing an external vel_x/vel_y bookkeeping system.
    """

    def __init__(self, cx, cy, process_noise=2.0, measurement_noise=30.0):
        # State: [cx, cy, vx, vy]
        self.x = np.array([cx, cy, 0.0, 0.0], dtype=float)

        # State transition matrix (constant-velocity model)
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)

        # Measurement matrix (we only observe position, not velocity)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        # Process noise covariance
        q = process_noise
        self.Q = np.diag([q, q, q * 0.5, q * 0.5])

        # Measurement noise covariance
        r = measurement_noise
        self.R = np.diag([r, r])

        # Initial state covariance — high uncertainty at start
        self.P = np.eye(4) * 500.0

    def predict(self):
        """Advance the state by one frame (no measurement)."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0], self.x[1]

    def update(self, cx_meas, cy_meas):
        """Incorporate a new YOLO measurement."""
        z = np.array([cx_meas, cy_meas])
        y = z - self.H @ self.x                      # innovation
        S = self.H @ self.P @ self.H.T + self.R      # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)     # Kalman gain
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
        """Hot-swap noise params (used to tighten/loosen during state changes)."""
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
    p.add_argument("--conf", type=float, default=0.25,
                   help="""YOLO detection confidence threshold.
LOWER (e.g. 0.10) = detects skier more aggressively, may cause false positives from spray/boat.
HIGHER (e.g. 0.25) = only confident detections, may lose skier in spray. (default: 0.25)""")

    p.add_argument("--jump_limit", type=float, default=0.12,
                   help="""Max fraction of frame width a new detection can be from last known position.
LOWER = stricter, rejects more false detections.
HIGHER = looser, follows fast moves better but risks bad detections. (default: 0.12)""")

    # ── Kalman noise ─────────────────────────────────────────────────────────
    p.add_argument("--process_noise", type=float, default=2.0,
                   help="""Kalman process noise — how much the filter trusts its own motion model.
LOWER (e.g. 0.5) = smoother, slower to change direction, coasts straighter.
HIGHER (e.g. 8.0) = adapts faster to direction changes, may be jitterier. (default: 2.0)""")

    p.add_argument("--measurement_noise", type=float, default=30.0,
                   help="""Kalman measurement noise — how much the filter trusts each YOLO detection.
LOWER (e.g. 10) = snappier, follows detections closely, more sensitive to jitter.
HIGHER (e.g. 80) = smoother, lags more behind real position. (default: 30.0)""")

    p.add_argument("--reacquire_noise", type=float, default=80.0,
                   help="""Measurement noise during REACQUIRING — higher than normal tracking
so the filter eases onto the skier rather than snapping. (default: 80.0)""")

    # ── Speed clamp (safety net — always applied) ────────────────────────────
    p.add_argument("--max_speed", type=int, default=120,
                   help="""Hard cap on pixels the crop center can move per frame (full-res 4K pixels).
Prevents ALL snapping regardless of detection quality — the last line of defense.
LOWER (e.g. 50) = smoother but may lag very fast skier.
HIGHER (e.g. 150) = follows faster, tiny risk of visible jump. (default: 120)""")

    # ── Lost-skier behavior ──────────────────────────────────────────────────
    p.add_argument("--min_reacquire", type=int, default=12,
                   help="""Frames to ignore new detections after losing skier.
Prevents immediately jumping to a wrong detection.
LOWER (e.g. 4) = re-locks sooner. HIGHER (e.g. 20) = coasts longer. (default: 12)""")

    p.add_argument("--reacquire_frames", type=int, default=5,
                   help="""Consecutive detections needed before switching back to full tracking.
LOWER = locks back on quickly. HIGHER = waits for more evidence. (default: 5)""")

    p.add_argument("--miss_grace", type=int, default=4,
                   help="""Consecutive missed detections required before leaving TRACKING
and entering COASTING. Prevents stutter from 1-3 frame flickers. (default: 4)""")

    # ── Framing ─────────────────────────────────────────────────────────────
    p.add_argument("--headroom", type=float, default=0.48,
                   help="""Vertical position of skier in frame (fraction from top of crop).
LOWER (e.g. 0.38) = more sky above skier.
HIGHER (e.g. 0.55) = skier higher in frame, more water below. (default: 0.48)""")

    p.add_argument("--output_height", type=int, default=1920,
                   help="Output height in pixels. Width auto-set to 9:16. (default: 1920)")

    p.add_argument("--model", type=str, default="yolov8s.pt")
    p.add_argument("--detect_width", type=int, default=1280)

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
    top_mask    = int(h * 0.12)
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

    # Reject boxes whose center is in the bottom 15% of the ROI (boat/gunwale zone)
    roi_bottom_y = bottom_mask
    y_cutoff = top_mask + (roi_bottom_y - top_mask) * 0.85
    boxes = [b for b in boxes if (b[1]+b[3])/2 < y_cutoff]
    if not boxes:
        return None

    # Size sanity filter — reject boxes wildly different from recent ground average
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

    AW = args.detect_width
    AH = int(AW * FH / FW); AH -= AH % 2
    sx = FW / AW
    sy = FH / AH
    jump_limit_px = args.jump_limit * AW

    print(f"Input : {FW}x{FH} @ {FPS}fps ({TOTAL} frames, {TOTAL/FPS:.1f}s)")
    print(f"Crop  : {CROP_W}x{CROP_H}  Output: {OUT_W}x{OUT_H}")
    print(f"Detect: {AW}x{AH}  Model: {args.model}")
    print(f"conf={args.conf}  jump_limit={args.jump_limit}  max_speed={args.max_speed}px/f")
    print(f"process_noise={args.process_noise}  measurement_noise={args.measurement_noise}")
    print(f"min_reacquire={args.min_reacquire}  reacquire_frames={args.reacquire_frames}")
    print(f"\nLoading YOLO ({args.model})...")
    model = YOLO(args.model)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, FPS, (OUT_W, OUT_H))

    # ── Kalman filter — initialised at frame center ──────────────────────────
    kf = KalmanTracker(
        cx=FW / 2.0,
        cy=FH * 0.45,
        process_noise=args.process_noise,
        measurement_noise=args.measurement_noise,
    )

    # smooth_cx/cy are the Kalman output — crop is placed here each frame
    smooth_cx, smooth_cy = kf.position

    last_cx_norm = 0.5

    # Detection averaging buffer — smooths YOLO jitter before Kalman update
    det_avg_window = 5
    det_buf_x = deque(maxlen=det_avg_window)
    det_buf_y = deque(maxlen=det_avg_window)

    # For confirmed-candidate logic during COASTING
    recent_dets = deque(maxlen=8)

    state             = "TRACKING"
    lost_frames       = 0
    reacq_frames      = 0
    miss_streak       = 0
    reacq_miss_streak = 0
    ground_box_heights = deque(maxlen=20)

    yolo_hits   = 0
    coast_count = 0
    reacq_count = 0

    print("Processing...\n")
    frame_num = 0

    # debug_log = open(args.output + ".debug.csv", "w")
    # debug_log.write("frame,time_s,state,detected,box_h,target_cx,target_cy,smooth_cx,smooth_cy,kf_vx,kf_vy\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        if frame_num % 60 == 0:
            pct = frame_num / TOTAL * 100
            vx, vy = kf.velocity
            print(f"  {frame_num}/{TOTAL} ({pct:.0f}%)  [{state}]  "
                  f"yolo={yolo_hits}  coast={coast_count}  reacq={reacq_count}  "
                  f"kf_vel=({vx:.1f},{vy:.1f})")

        small      = cv2.resize(frame, (AW, AH))
        expected_h = float(np.mean(ground_box_heights)) if len(ground_box_heights) >= 5 else None
        result     = detect_skier_yolo(model, small, args.conf, last_cx_norm, jump_limit_px, expected_h)

        # ── Always predict first — this is what makes coasting work for free ─
        kf.predict()

        if state == "TRACKING":
            if result is not None:
                rcx, rcy, box_h = result
                raw_cx = rcx * sx
                raw_cy = rcy * sy

                det_buf_x.append(raw_cx)
                det_buf_y.append(raw_cy)
                det_cx = float(np.mean(det_buf_x))
                det_cy = float(np.mean(det_buf_y))

                # Update Kalman with averaged detection
                kf.update(det_cx, det_cy)

                ground_box_heights.append(box_h)
                recent_dets.append((frame_num, det_cx, det_cy))
                last_cx_norm = rcx / AW
                lost_frames  = 0
                miss_streak  = 0
                yolo_hits   += 1

            else:
                miss_streak += 1
                if miss_streak >= args.miss_grace:
                    # Genuinely lost — switch to COASTING
                    # Kalman predict (already called above) handles momentum
                    state        = "COASTING"
                    lost_frames  = 1
                    coast_count += 1
                # else: brief flicker, just hold — Kalman predicts naturally

        elif state == "COASTING":
            coast_count += 1
            lost_frames += 1
            # Kalman predict (already called) advances position by velocity estimate.
            # No manual vel_x/vel_y needed.

            if result is not None:
                rcx, rcy, box_h = result
                cand_cx = rcx * sx
                cand_cy = rcy * sy

                # Trust single-frame detections that are close to the predicted
                # trajectory; require confirmation for larger jumps
                pred_cx, pred_cy = kf.position
                dist_from_pred = np.hypot(cand_cx - pred_cx, cand_cy - pred_cy)
                max_plausible_jump = AW * sx * 0.18

                if dist_from_pred <= max_plausible_jump:
                    kf.update(cand_cx, cand_cy)
                    last_cx_norm = rcx / AW

                # Transition to REACQUIRING after min_reacquire coast frames
                if lost_frames >= args.min_reacquire:
                    recent_dets.clear()
                    det_buf_x.clear()
                    det_buf_y.clear()
                    recent_dets.append((frame_num, cand_cx, cand_cy))
                    state        = "REACQUIRING"
                    reacq_frames = 0
                    reacq_miss_streak = 0
                    last_cx_norm = rcx / AW
                    # Increase measurement noise so entry is gradual, not a snap
                    kf.set_noise(measurement_noise=args.reacquire_noise)

        elif state == "REACQUIRING":
            reacq_count += 1

            if result is not None:
                rcx, rcy, box_h = result
                det_cx = rcx * sx
                det_cy = rcy * sy

                kf.update(det_cx, det_cy)
                last_cx_norm = rcx / AW
                ground_box_heights.append(box_h)
                recent_dets.append((frame_num, det_cx, det_cy))
                reacq_frames      += 1
                reacq_miss_streak  = 0

                if reacq_frames >= args.reacquire_frames:
                    # Confirmed re-lock — restore normal measurement noise
                    kf.set_noise(measurement_noise=args.measurement_noise)
                    state       = "TRACKING"
                    miss_streak = 0

            else:
                reacq_miss_streak += 1
                if reacq_miss_streak >= 2:
                    # Lost again mid-reacquire — fall back to COASTING
                    state        = "COASTING"
                    lost_frames += reacq_frames
                    reacq_miss_streak = 0
                    kf.set_noise(measurement_noise=args.measurement_noise)
                # else: hold, Kalman predicts forward naturally

        # ── Read Kalman output position ───────────────────────────────────────
        new_cx, new_cy = kf.position

        # ── HARD SPEED CLAMP — final safety net, nothing bypasses this ────────
        move_x = new_cx - smooth_cx
        move_y = new_cy - smooth_cy
        dist   = np.hypot(move_x, move_y)
        if dist > args.max_speed:
            f       = args.max_speed / dist
            move_x *= f
            move_y *= f

        smooth_cx += move_x
        smooth_cy += move_y

        # Sync Kalman state to clamped position so it doesn't drift away
        kf.x[0] = smooth_cx
        kf.x[1] = smooth_cy

        box_h_log = result[2] if result is not None else -1
        vx, vy = kf.velocity
        # debug_log.write(f"{frame_num},{frame_num/FPS:.3f},{state},{result is not None},{box_h_log:.1f},{new_cx:.1f},{new_cy:.1f},{smooth_cx:.1f},{smooth_cy:.1f},{vx:.2f},{vy:.2f}\n")

        x1, y1 = clamp_crop(smooth_cx, smooth_cy, CROP_W, CROP_H, FW, FH, args.headroom)
        cropped = frame[y1:y1+CROP_H, x1:x1+CROP_W]
        resized = cv2.resize(cropped, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
        writer.write(resized)

    cap.release()
    writer.release()
    # debug_log.close()

    print(f"\nDone → {args.output}")
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
