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

    p.add_argument("--reacquire_max_speed", type=int, default=30,
                   help="""Hard per-frame pixel cap while REACQUIRING.
Lower than max_speed so the catch-up after a long coast is visually gradual
rather than a sudden slide at full speed. (default: 30)""")

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


def detect_skier_yolo(model, frame_small, conf, search_cx_norm, search_radius_px,
                       expected_box_h=None, is_tracking=True):
    """
    Returns (cx, cy, box_height, confidence) in frame_small coords, or None.

    search_cx_norm   — normalized [0,1] x-center to search around.
                       In TRACKING: last detected position.
                       In COASTING/REACQUIRING: Kalman predicted position,
                       which is where the skier *should* be, not where they
                       were last seen.
    search_radius_px — how far from search_cx_norm to accept a detection.
                       Tight (0.12*w) while TRACKING; widens progressively
                       during long coasts so the skier can be found anywhere
                       in the frame after a multi-second gap.
    is_tracking      — True in TRACKING/AIRBORNE: use tight size filter (0.45x)
                       to block buoys. False in COASTING/REACQUIRING: loosen to
                       0.25x so a distant/small skier isn't rejected by a rolling
                       average built from close-up ground-phase frames.
    """
    h, w = frame_small.shape[:2]
    top_mask    = int(h * 0.06)
    bottom_mask = int(h * 0.65)
    roi = frame_small[top_mask:bottom_mask, :]

    results = model(roi, classes=[0], verbose=False, conf=conf)
    boxes = []
    for r in results:
        for box in r.boxes:
            b = box.xyxy[0].cpu().numpy().copy()
            b[1] += top_mask
            b[3] += top_mask
            c = float(box.conf[0].cpu().numpy())
            boxes.append((b, c))

    if not boxes:
        return None

    # Y-zone filter — reject center-of-box in boat/gunwale zone
    y_cutoff = top_mask + (bottom_mask - top_mask) * 0.85
    boxes = [(b, c) for b, c in boxes if (b[1]+b[3])/2 < y_cutoff]
    if not boxes:
        return None

    # ── Hard minimum box height ──────────────────────────────────────────────
    # Skier is typically 40-100px tall at 1280px detect width.
    # Buoy in test footage was 20-27px — floor kills it outright.
    boxes = [(b, c) for b, c in boxes if (b[3]-b[1]) >= 32]
    if not boxes:
        return None

    # ── Size sanity filter ───────────────────────────────────────────────────
    # TRACKING: tight 0.45x lower bound blocks buoys (27px buoy vs 50px avg).
    # COASTING/REACQUIRING: loosen to 0.25x — the rolling average is built from
    # close-up frames where box_h ~90-100px. A distant skier in spray may have
    # box_h ~40px which is < 90*0.45=40.5 and gets wrongly rejected. During
    # coast we care more about finding the skier than blocking false positives
    # (the proximity/aspect/min-height filters still protect us).
    size_lower = 0.45 if is_tracking else 0.25
    if expected_box_h is not None and expected_box_h > 0:
        boxes = [(b, c) for b, c in boxes
                 if (b[3]-b[1]) < expected_box_h * 2.2
                 and (b[3]-b[1]) > expected_box_h * size_lower]
        if not boxes:
            return None

    # ── Aspect ratio filter ──────────────────────────────────────────────────
    # Skier is always taller than wide. Buoys/balls are square or wider.
    boxes = [(b, c) for b, c in boxes if (b[3]-b[1]) > (b[2]-b[0]) * 0.92]
    if not boxes:
        return None

    # ── Proximity filter ─────────────────────────────────────────────────────
    # Use search_cx_norm (Kalman predicted position during coast) as the
    # center, with a radius that widens the longer we've been lost.
    # When search_radius_px >= w the filter is effectively disabled (full scan).
    if search_cx_norm is not None and search_radius_px < w:
        search_cx_px = search_cx_norm * w
        filtered = [(b, c) for b, c in boxes
                    if abs((b[0]+b[2])/2 - search_cx_px) <= search_radius_px]
        if filtered:
            boxes = filtered
        # If nothing within radius, fall through to full-frame best pick
        # rather than returning None — this is the key change. During a long
        # coast we'd rather find the skier anywhere than miss entirely.

    # ── Confidence-weighted height score ─────────────────────────────────────
    best, best_conf = max(boxes, key=lambda bc: (bc[0][3]-bc[0][1]) * bc[1])
    cx    = float((best[0]+best[2])/2)
    cy    = float((best[1]+best[3])/2)
    box_h = float(best[3]-best[1])
    return cx, cy, box_h, best_conf


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

    # ── Cold-start: seed Kalman from first detectable frame ─────────────────
    # Initialising at frame center caused huge velocity spikes in frames 1-13
    # as the filter tried to chase the skier (who was already mid-frame).
    # Instead, scan the first few frames to find the skier, then seed directly.
    print("Seeding tracker from first frame...")
    seed_cx, seed_cy = FW / 2.0, FH * 0.42   # fallback if no detection
    for _seed_attempt in range(30):            # scan up to first 30 frames
        ret_s, frame_s = cap.read()
        if not ret_s:
            break
        small_s = cv2.resize(frame_s, (AW, AH))
        seed_result = detect_skier_yolo(model, small_s, args.conf, 0.5, jump_limit_px)
        if seed_result is not None:
            rcx_s, rcy_s, _, _ = seed_result
            seed_cx = rcx_s * sx
            seed_cy = rcy_s * sy
            print(f"  Skier found at frame {_seed_attempt+1}: cx={seed_cx:.0f} cy={seed_cy:.0f}")
            break
    else:
        print("  No skier found in first 30 frames — using frame center")
    # Rewind so we re-process from frame 1 (VideoCapture supports this)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    kf = KalmanTracker(
        cx=seed_cx,
        cy=seed_cy,
        process_noise=args.process_noise,
        measurement_noise=args.measurement_noise,
    )

    smooth_cx, smooth_cy = kf.position
    last_cx_norm = seed_cx / FW

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
    reacq_speed_cap   = 30.0  # set on REACQUIRING entry from actual gap size
    reacq_ramp_frames = 0     # counts frames since last reacquire for speed ramp-up

    yolo_hits   = 0
    coast_count = 0
    reacq_count = 0
    air_count   = 0

    print("Processing...\n")
    frame_num = 0

    debug_log = open(args.output + ".debug.csv", "w")
    debug_log.write("frame,time_s,state,detected,box_h,conf,is_airborne,target_cx,target_cy,smooth_cx,smooth_cy,kf_vx,kf_vy\n")

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

        # ── State-aware search center and radius ─────────────────────────────
        # TRACKING: tight window around last detected position — false positive
        #   rejection is most important here.
        # COASTING/REACQUIRING: use Kalman predicted position as search center
        #   (better than last_cx_norm which is stale), and widen the radius
        #   progressively so a skier who has moved far can still be found.
        #   After ~5s the whole frame is searched — we'd rather risk one false
        #   positive than never reacquire.
        if state == "TRACKING" or state == "AIRBORNE":
            search_cx  = last_cx_norm
            search_rad = jump_limit_px
        else:
            # Kalman predicted x in detect-frame coords
            pred_cx_full, _ = kf.position
            search_cx  = pred_cx_full / (FW / AW) / AW   # → [0,1] in detect space
            search_cx  = float(np.clip(search_cx, 0.0, 1.0))
            # Widen by 8px/second lost, floor at jump_limit_px, ceil at full width
            widen      = (lost_frames / FPS) * 8.0 * (AW / 1280.0)
            search_rad = min(AW, jump_limit_px + widen)

        is_tracking_state = state in ("TRACKING", "AIRBORNE")
        result = detect_skier_yolo(model, small, args.conf, search_cx, search_rad,
                                   expected_h, is_tracking=is_tracking_state)

        # ── Airborne heuristic ───────────────────────────────────────────────
        is_airborne_detection = False
        if result is not None:
            _, _, box_h, det_conf = result
            if len(ground_box_heights) >= 8:
                avg_ground_h = np.mean(ground_box_heights)
                if box_h < avg_ground_h * 0.60:
                    is_airborne_detection = True

        # ── Always predict first ─────────────────────────────────────────────
        kf.predict()

        # ════════════════════════════════════════════════════════════════════
        if state == "TRACKING":
            if result is not None:
                rcx, rcy, box_h, det_conf = result
                det_cx = rcx * sx
                det_cy = rcy * sy

                # ── Plausibility gate in TRACKING (fix #2) ───────────────────
                # The buoy was slamming the crop while still in TRACKING because
                # there was no distance check here — only COASTING had one.
                # Reject any detection that's implausibly far from the Kalman
                # prediction once we have a stable rolling average.
                pred_cx, pred_cy = kf.position
                dist_from_pred = np.hypot(det_cx - pred_cx, det_cy - pred_cy)
                if dist_from_pred > args.max_speed * 2 and len(ground_box_heights) >= 5:
                    result = None  # treat as missed frame — don't update Kalman
                elif is_airborne_detection:
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
                    # Fix #3: only add high-confidence detections to rolling avg
                    # so it never gets poisoned by marginal detections
                    if det_conf >= 0.4:
                        ground_box_heights.append(box_h)
                    recent_dets.append((frame_num, det_cx, det_cy))
                    last_cx_norm = rcx / AW
                    lost_frames  = 0
                    miss_streak  = 0
                    yolo_hits   += 1
            if result is None:
                miss_streak += 1
                # If box_h has been shrinking toward airborne range the skier
                # is probably on the ramp with intermittent detections.
                # Double miss_grace to prevent rapid TRACKING/COASTING/REACQUIRING
                # cycling that causes jitter during the ramp phase.
                if (last_det_box_h is not None and len(ground_box_heights) >= 5
                        and last_det_box_h < float(np.mean(ground_box_heights)) * 0.75):
                    effective_grace = args.miss_grace * 2
                else:
                    effective_grace = args.miss_grace
                if miss_streak >= effective_grace:
                    state             = "COASTING"
                    lost_frames       = 1
                    coast_count      += 1
                    pending_candidate = None
                    reacq_ramp_frames = 0

        elif state == "AIRBORNE":
            air_count += 1
            if result is not None:
                rcx, rcy, box_h, det_conf = result
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
                    if box_h >= avg_ground_h * 0.80:
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

            # Decay Kalman velocity each frame while coasting — without this the
            # constant-velocity model flies off-screen indefinitely at whatever
            # velocity it had when tracking was lost. Same role as the old
            # vel_x *= decay + coast_scale combination.
            kf.x[2] *= 0.92 * 0.75   # decay * coast_scale
            kf.x[3] *= 0.92 * 0.75

            if result is not None:
                rcx, rcy, box_h, det_conf = result
                cand_cx = rcx * sx
                cand_cy = rcy * sy

                pred_cx, pred_cy = kf.position
                dist_from_pred   = np.hypot(cand_cx - pred_cx, cand_cy - pred_cy)
                max_plausible    = AW * sx * 0.18

                # Only update Kalman during coast if detection is both close
                # to predicted position AND has reasonable confidence.
                # Low-conf detections (conf<0.35) during coast were causing
                # 54px jumps (frame 663: conf=0.15 moved crop 54px sideways).
                if dist_from_pred <= max_plausible and det_conf >= 0.35:
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
                            kf.x[2] = 0.0
                            kf.x[3] = 0.0
                            # ── Gap-based speed cap (the real fix) ───────────
                            # Previous formula used lost_frames which is 25x
                            # too permissive (CSV showed 69px gap closed in
                            # 0.10s instead of 2.5s).
                            # Correct approach: measure the actual pixel
                            # distance from current crop to the detected skier,
                            # then set cap so it takes exactly 2.5s to close.
                            # cap = gap_px / (2.5 * FPS)
                            # Floor at 1.5px/frame so it never fully stalls.
                            gap_px = np.hypot(cand_cx - smooth_cx,
                                              cand_cy - smooth_cy)
                            reacq_speed_cap = max(1.5, gap_px / (2.5 * FPS))
                            # Scale measurement noise from gap size too —
                            # large gap = Kalman should barely react at first.
                            # gap/10 gives ~7 for a 69px gap, ~30 for 300px.
                            scaled_noise = args.reacquire_noise + (gap_px / 10.0)
                            kf.set_noise(measurement_noise=scaled_noise)
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
                rcx, rcy, box_h, det_conf = result
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
                    state             = "TRACKING"
                    miss_streak       = 0
                    reacq_ramp_frames = 1   # start speed ramp — don't snap to max_speed
            else:
                reacq_miss_streak += 1
                if reacq_miss_streak >= 2:
                    state           = "COASTING"
                    lost_frames    += reacq_frames
                    reacq_miss_streak = 0
                    reacq_speed_cap = args.reacquire_max_speed  # reset for next entry
                    kf.set_noise(measurement_noise=args.measurement_noise)

        # ── Read Kalman position ─────────────────────────────────────────────
        new_cx, new_cy = kf.position

        # ── HARD SPEED CLAMP — always applied, nothing bypasses this ─────────
        if state == "AIRBORNE":
            cap_speed = args.air_max_speed
        elif state == "COASTING":
            # During coast, cap movement at 15px/frame even for valid
            # Kalman corrections. The 55px max_speed was firing on coast
            # detections and creating visible hops (frames 276,692,1187).
            cap_speed = 15.0
        elif state == "REACQUIRING":
            # Gap-based cap: gap_px / (2.5 * FPS) → always 2.5s to close.
            cap_speed = reacq_speed_cap
        elif state == "TRACKING" and reacq_ramp_frames > 0:
            # Ramp speed cap from reacq_speed_cap up to max_speed over 1 second
            # after a reacquire. Prevents the snap from 1.5px/f → 55px/f on
            # the exact frame we transition from REACQUIRING to TRACKING.
            ramp_progress = min(1.0, reacq_ramp_frames / FPS)
            cap_speed = reacq_speed_cap + (args.max_speed - reacq_speed_cap) * ramp_progress
            reacq_ramp_frames += 1
            if ramp_progress >= 1.0:
                reacq_ramp_frames = 0  # ramp complete
        else:
            cap_speed = args.max_speed
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

        box_h_log  = result[2] if result is not None else -1
        conf_log   = result[3] if result is not None else -1
        vx, vy = kf.velocity
        debug_log.write(f"{frame_num},{frame_num/FPS:.3f},{state},{result is not None},{box_h_log:.1f},{conf_log:.2f},{is_airborne_detection},{new_cx:.1f},{new_cy:.1f},{smooth_cx:.1f},{smooth_cy:.1f},{vx:.2f},{vy:.2f}\n")

        x1, y1 = clamp_crop(smooth_cx, smooth_cy, CROP_W, CROP_H, FW, FH, args.headroom)
        cropped = frame[y1:y1+CROP_H, x1:x1+CROP_W]
        resized = cv2.resize(cropped, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
        writer.write(resized)

    cap.release()
    writer.release()
    debug_log.close()

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
