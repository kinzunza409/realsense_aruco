import matplotlib
matplotlib.use("Agg")

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


LM = mp.solutions.pose.PoseLandmark   # convenient alias

# The subset of landmarks we care about
UPPER_LIMB_IDS = {
    LM.LEFT_SHOULDER,
    LM.RIGHT_SHOULDER,
    LM.LEFT_ELBOW,
    LM.RIGHT_ELBOW,
    LM.LEFT_WRIST,
    LM.RIGHT_WRIST,
    LM.LEFT_PINKY,
    LM.RIGHT_PINKY,
    LM.LEFT_INDEX,
    LM.RIGHT_INDEX,
    LM.LEFT_THUMB,
    LM.RIGHT_THUMB,
}

# Bone segments to draw
UPPER_LIMB_CONNECTIONS = [
    (LM.LEFT_SHOULDER,  LM.LEFT_ELBOW),
    (LM.LEFT_ELBOW,     LM.LEFT_WRIST),
    (LM.LEFT_WRIST,     LM.LEFT_PINKY),
    (LM.LEFT_WRIST,     LM.LEFT_INDEX),
    (LM.LEFT_WRIST,     LM.LEFT_THUMB),
    (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW),
    (LM.RIGHT_ELBOW,    LM.RIGHT_WRIST),
    (LM.RIGHT_WRIST,    LM.RIGHT_PINKY),
    (LM.RIGHT_WRIST,    LM.RIGHT_INDEX),
    (LM.RIGHT_WRIST,    LM.RIGHT_THUMB),
    (LM.LEFT_SHOULDER,  LM.RIGHT_SHOULDER),  # shoulder bar
]

# Joint angle definitions: (label, point_a, vertex, point_b)
# Angle is measured AT vertex between rays (vertex -> a) and (vertex -> b)
JOINT_ANGLE_DEFS = [
    ("L_elbow",    LM.LEFT_SHOULDER,    LM.LEFT_ELBOW,     LM.LEFT_WRIST),
    ("R_elbow",    LM.RIGHT_SHOULDER,   LM.RIGHT_ELBOW,    LM.RIGHT_WRIST),
    ("L_shoulder", LM.LEFT_ELBOW,       LM.LEFT_SHOULDER,  LM.RIGHT_SHOULDER),
    ("R_shoulder", LM.RIGHT_ELBOW,      LM.RIGHT_SHOULDER, LM.LEFT_SHOULDER),
    ("L_wrist",    LM.LEFT_ELBOW,       LM.LEFT_WRIST,     LM.LEFT_INDEX),
    ("R_wrist",    LM.RIGHT_ELBOW,      LM.RIGHT_WRIST,    LM.RIGHT_INDEX),
]

# Colours (BGR)
COL_LANDMARK   = (0, 255, 120)   # bright green
COL_BONE       = (255, 200, 0)   # gold / cyan
COL_LOW_CONF   = (80, 80, 80)    # grey for occluded landmarks
COL_TEXT_BG    = (20, 20, 20)
COL_TEXT_FG    = (220, 220, 220)

VISIBILITY_THRESHOLD = 0.5  # landmarks below this are treated as occluded


def angle_between(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> float:
    """Angle in degrees at `vertex` between vectors (vertex->a) and (vertex->b)."""
    va = a - vertex
    vb = b - vertex
    n_a = np.linalg.norm(va)
    n_b = np.linalg.norm(vb)
    if n_a < 1e-6 or n_b < 1e-6:
        return float("nan")
    cos_t = np.clip(np.dot(va, vb) / (n_a * n_b), -1.0, 1.0)
    return math.degrees(math.acos(cos_t))


def lm_to_px(lm_obj, w: int, h: int):
    """Normalised landmark -> pixel (x, y)."""
    return int(lm_obj.x * w), int(lm_obj.y * h)


def lm_to_2d(lm_obj) -> np.ndarray:
    """Normalised landmark -> float32 (x, y) array."""
    return np.array([lm_obj.x, lm_obj.y], dtype=np.float32)


def draw_skeleton(frame, landmarks, w: int, h: int):
    lm_list = landmarks.landmark

    for (id_a, id_b) in UPPER_LIMB_CONNECTIONS:
        la, lb = lm_list[id_a], lm_list[id_b]
        if la.visibility < VISIBILITY_THRESHOLD or lb.visibility < VISIBILITY_THRESHOLD:
            continue
        cv2.line(frame, lm_to_px(la, w, h), lm_to_px(lb, w, h),
                 COL_BONE, 2, cv2.LINE_AA)

    for lm_id in UPPER_LIMB_IDS:
        lm_obj = lm_list[lm_id]
        px = lm_to_px(lm_obj, w, h)
        colour = COL_LANDMARK if lm_obj.visibility >= VISIBILITY_THRESHOLD else COL_LOW_CONF
        cv2.circle(frame, px, 5, colour, -1, cv2.LINE_AA)
        cv2.circle(frame, px, 6, (0, 0, 0), 1, cv2.LINE_AA)  # outline


def draw_angle_labels(frame, landmarks, angles: dict, w: int, h: int):
    """Draw each angle value next to its vertex landmark in the frame."""
    lm_list = landmarks.landmark
    for (name, _id_a, id_vertex, _id_b) in JOINT_ANGLE_DEFS:
        val = angles.get(name, float("nan"))
        if math.isnan(val):
            continue
        lm_obj = lm_list[id_vertex]
        if lm_obj.visibility < VISIBILITY_THRESHOLD:
            continue
        px, py = lm_to_px(lm_obj, w, h)
        label = f"{val:.1f}deg"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(frame, (px + 8, py - th - 4), (px + 12 + tw, py + 4),
                      COL_TEXT_BG, -1)
        cv2.putText(frame, label, (px + 10, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_TEXT_FG, 1, cv2.LINE_AA)


def draw_angle_panel(frame, angles: dict):
    """Top-left corner summary panel."""
    x, y = 10, 22
    for name, val in angles.items():
        text = f"{name}: {val:.1f}deg" if not math.isnan(val) else f"{name}:   --"
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, COL_TEXT_FG, 1, cv2.LINE_AA)
        y += 20


def draw_hud(frame, fps: float, paused: bool, h: int):
    status = "PAUSED" if paused else f"FPS {fps:.1f}"
    cv2.putText(frame, status, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 255, 80), 1, cv2.LINE_AA)

_first_log = True

def log_angles_terminal(angles: dict, frame_idx: int):
    """Overwrite the same terminal lines each frame for a clean readout."""
    global _first_log
    lines = [f"  Frame {frame_idx:>7d}"]
    for name, val in angles.items():
        val_str = f"{val:6.1f} deg" if not math.isnan(val) else "    --    "
        lines.append(f"    {name:<14s} {val_str}")

    if not _first_log:
        # Move cursor up N lines to overwrite
        sys.stdout.write(f"\033[{len(lines)}A")
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()
    _first_log = False


def compute_angles(landmarks) -> dict:
    lm_list = landmarks.landmark
    angles = {}
    for (name, id_a, id_v, id_b) in JOINT_ANGLE_DEFS:
        la, lv, lb = lm_list[id_a], lm_list[id_v], lm_list[id_b]
        if min(la.visibility, lv.visibility, lb.visibility) < VISIBILITY_THRESHOLD:
            angles[name] = float("nan")
        else:
            # Use 2-D (x, y) — more stable on a standard monocular webcam.
            # MediaPipe's z is estimated relative depth; it's noisier than x/y
            # for joint-angle purposes unless you have a calibrated depth stream.
            angles[name] = angle_between(lm_to_2d(la), lm_to_2d(lv), lm_to_2d(lb))
    return angles


def run(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open source: {source!r}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Source opened  {w}x{h}")
    print(f"[INFO] Controls:  q/ESC=quit   p=pause   s=save-frame\n")

    # model_complexity=1 is a solid balance between accuracy and speed on a
    # modern laptop GPU / CPU.  Use 0 if you need more FPS, 2 for max accuracy.
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    paused    = False
    frame_idx = 0
    fps_ema   = 0.0          # exponential moving average for display
    t_prev    = time.perf_counter()
    save_dir  = Path("saved_frames")
    frame     = None         # keep a reference so 's' can save after any frame

    try:
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                paused = not paused
                print(f"\n[INFO] {'Paused' if paused else 'Resumed'}")
            if key == ord("s") and frame is not None:
                save_dir.mkdir(exist_ok=True)
                fname = save_dir / f"frame_{frame_idx:06d}.png"
                cv2.imwrite(str(fname), frame)
                print(f"\n[INFO] Saved -> {fname}")

            if paused:
                continue

            ret, frame = cap.read()
            if not ret:
                print("\n[INFO] End of stream.")
                break

            # FPS (exponential smoothing, alpha=0.1 -> stable display)
            t_now  = time.perf_counter()
            dt     = max(t_now - t_prev, 1e-6)
            t_prev = t_now
            fps_ema = fps_ema * 0.9 + (1.0 / dt) * 0.1

            # Inference (MediaPipe expects RGB)
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if result.pose_landmarks:
                angles = compute_angles(result.pose_landmarks)
                draw_skeleton(frame, result.pose_landmarks, w, h)
                draw_angle_labels(frame, result.pose_landmarks, angles, w, h)
                draw_angle_panel(frame, angles)
                log_angles_terminal(angles, frame_idx)
            else:
                cv2.putText(frame, "No person detected", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 60, 255), 2)

            draw_hud(frame, fps_ema, paused, h)
            cv2.imshow("Upper Limb Pose", frame)
            frame_idx += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        print("\n[INFO] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediaPipe upper-limb pose detector")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--camera", type=int, default=0, metavar="INDEX",
                       help="Webcam device index (default: 0)")
    group.add_argument("--video", type=str, metavar="PATH",
                       help="Path to a video file instead of a webcam")
    args = parser.parse_args()

    run(args.video if args.video else args.camera)