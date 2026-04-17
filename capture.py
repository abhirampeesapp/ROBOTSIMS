import os
import sys
import time
import threading
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# ROS 2 — guarded import
# ─────────────────────────────────────────────────────────────────────────────
try:
    import rclpy                                          # type: ignore[import]
    from rclpy.node import Node                           # type: ignore[import]
    from rclpy.executors import MultiThreadedExecutor     # type: ignore[import]
    from rclpy.qos import (                               # type: ignore[import]
        QoSProfile, ReliabilityPolicy, HistoryPolicy
    )
    from sensor_msgs.msg import Image                     # type: ignore[import]
    from cv_bridge import CvBridge, CvBridgeError         # type: ignore[import]
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    class Node:
        def __init__(self, *a, **kw): pass
        def create_subscription(self, *a, **kw): pass
        def create_timer(self, *a, **kw): pass
        def declare_parameter(self, *a, **kw): pass
        def get_parameter(self, name):
            class _P:
                value = None
            return _P()
        def get_logger(self): return logging.getLogger("capture.stub")
        def destroy_node(self): pass

from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("capture")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    "topic":          os.environ.get("CAM_TOPIC",    "/camera/image_raw"),
    "model":          os.environ.get("YOLO_MODEL",   "yolov8n.pt"),
    "conf":           float(os.environ.get("YOLO_CONF",    "0.5")),
    "save_dir":       Path(os.environ.get("SAVE_DIR",      "captures")),
    "mode":           os.environ.get("CAPTURE_MODE",  "continuous"),
    "show_window":    False,          # ← Flask handles display now; window disabled
    "save_annotated": os.environ.get("SAVE_ANNOT",   "1") == "1",
    "save_raw":       os.environ.get("SAVE_RAW",     "1") == "1",
    "skip_frames":    int(os.environ.get("YOLO_SKIP",     "1")),
    "min_crop_px":    int(os.environ.get("MIN_CROP",      "1600")),
    "queue_depth":    int(os.environ.get("QOS_DEPTH",     "10")),
}

CFG["save_dir"].mkdir(parents=True, exist_ok=True)
(CFG["save_dir"] / "annotated").mkdir(exist_ok=True)
(CFG["save_dir"] / "raw").mkdir(exist_ok=True)
(CFG["save_dir"] / "crops").mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SHARED FRAME  ← NEW
# Flask reads this; CaptureNode / WebcamCapture writes it.
# Protected by a threading.Lock so there are no torn reads.
# ─────────────────────────────────────────────────────────────────────────────
shared_frame: np.ndarray | None = None
shared_frame_lock = threading.Lock()


def set_shared_frame(frame: np.ndarray) -> None:
    """Thread-safe write — called from callback / webcam loop."""
    global shared_frame
    with shared_frame_lock:
        shared_frame = frame.copy()


def get_shared_frame() -> np.ndarray | None:
    """Thread-safe read — called from Flask stream generator."""
    with shared_frame_lock:
        return shared_frame.copy() if shared_frame is not None else None


# ─────────────────────────────────────────────────────────────────────────────
# YOLO
# ─────────────────────────────────────────────────────────────────────────────
log.info(f"Loading YOLO model: {CFG['model']}")
try:
    yolo = YOLO(CFG["model"])
    yolo(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)   # warm-up
    log.info("YOLO ready ✓")
except Exception as exc:
    log.critical(f"Failed to load YOLO model: {exc}")
    sys.exit(1)

_yolo_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION RESULT
# ─────────────────────────────────────────────────────────────────────────────
class DetectionResult:
    def __init__(self, raw_results, names):
        self.boxes: list[dict] = []
        for r in raw_results:
            for box in r.boxes:
                self.boxes.append({
                    "label": names[int(box.cls[0])],
                    "conf":  round(float(box.conf[0]), 3),
                    "xyxy":  [int(v) for v in box.xyxy[0].tolist()],
                })

    @property
    def count(self): return len(self.boxes)

    @property
    def labels(self): return [b["label"] for b in self.boxes]

    def __repr__(self):
        return f"DetectionResult({self.count} objects: {self.labels})"


# ─────────────────────────────────────────────────────────────────────────────
# YOLO RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def run_yolo(frame: np.ndarray) -> tuple[np.ndarray, DetectionResult]:
    with _yolo_lock:
        results = yolo(frame, conf=CFG["conf"], verbose=False)
    det       = DetectionResult(results, yolo.names)
    annotated = results[0].plot()
    return annotated, det


# ─────────────────────────────────────────────────────────────────────────────
# FILE SAVER
# ─────────────────────────────────────────────────────────────────────────────
class FrameSaver:
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self._count   = 0

    def _ts(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    def save(self, frame: np.ndarray, annotated: np.ndarray,
             det: DetectionResult) -> dict:
        ts    = self._ts()
        saved: dict = {}
        self._count += 1

        if CFG["save_raw"]:
            p = self.save_dir / "raw" / f"raw_{ts}.jpg"
            cv2.imwrite(str(p), frame)
            saved["raw"] = str(p)

        if CFG["save_annotated"]:
            p = self.save_dir / "annotated" / f"annot_{ts}.jpg"
            cv2.imwrite(str(p), annotated)
            saved["annotated"] = str(p)

        for box in det.boxes:
            x1, y1, x2, y2 = box["xyxy"]
            crop = frame[y1:y2, x1:x2]
            if crop.size >= CFG["min_crop_px"]:
                p = self.save_dir / "crops" / f"{box['label']}_{ts}_crop.jpg"
                cv2.imwrite(str(p), crop)
                saved.setdefault("crops", []).append(str(p))

        return saved


# ─────────────────────────────────────────────────────────────────────────────
# HUD DRAWING  (shared between ROS and webcam paths)
# ─────────────────────────────────────────────────────────────────────────────
def draw_hud(frame: np.ndarray, det: DetectionResult,
             source: str, fps: float, save_count: int) -> None:
    """Draw overlay onto *frame* in-place. No imshow calls."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 22), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"{source}  MODE:{CFG['mode'].upper()}  FPS:{fps:.1f}  "
        f"CONF:{CFG['conf']:.2f}  SAVED:{save_count}",
        (6, 15), font, 0.42, (0, 220, 140), 1,
    )

    # Detection badge (top-right)
    badge_color = (30, 30, 160) if det.count == 0 else (20, 140, 20)
    cv2.rectangle(frame, (w - 110, 0), (w, 22), badge_color, -1)
    cv2.putText(frame, f"DET: {det.count}", (w - 104, 15),
                font, 0.5, (255, 255, 255), 1)

    # Label strip (bottom)
    if det.count > 0:
        label_str = "  ".join(
            f"{b['label']} {b['conf']:.0%}" for b in det.boxes[:6]
        )
        cv2.rectangle(frame, (0, h - 20), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, label_str, (6, h - 6),
                    font, 0.42, (80, 220, 160), 1)

    # Corner brackets
    c, L = (0, 200, 120), 18
    for (x, y) in [(0, 0), (w, 0), (0, h), (w, h)]:
        sx = 1 if x == 0 else -1
        sy = 1 if y == 0 else -1
        cv2.line(frame, (x, y), (x + sx * L, y), c, 1)
        cv2.line(frame, (x, y), (x, y + sy * L), c, 1)

    # Crosshair
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - 20, cy), (cx - 8, cy), c, 1)
    cv2.line(frame, (cx + 8,  cy), (cx + 20, cy), c, 1)
    cv2.line(frame, (cx, cy - 20), (cx, cy - 8), c, 1)
    cv2.line(frame, (cx, cy + 8),  (cx, cy + 20), c, 1)
    cv2.circle(frame, (cx, cy), 4, c, 1)


# ─────────────────────────────────────────────────────────────────────────────
# WEBCAM FALLBACK  (no ROS — writes to shared_frame for Flask)
# ─────────────────────────────────────────────────────────────────────────────
class WebcamCapture:
    """
    Standalone capture loop using a local webcam.
    Writes annotated frames to shared_frame so Flask can stream them.
    cv2.imshow is NOT used — Flask is the display layer.
    """

    def __init__(self):
        self._saver       = FrameSaver(CFG["save_dir"])
        self._save_count  = 0
        self._frame_count = 0
        self._fps_timer   = time.time()
        self._fps         = 0.0
        self._cap         = self._open_camera()

    # ── camera open ──────────────────────────────────────────────────────────
    def _open_camera(self) -> cv2.VideoCapture | None:
        for idx in range(4):
            for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        log.info(f"Webcam opened: index={idx} ✓")
                        return cap
                cap.release()
        log.error("No webcam found.")
        return None

    # ── main loop ─────────────────────────────────────────────────────────────
    def run(self) -> None:
        if self._cap is None:
            log.critical("No camera available — exiting.")
            sys.exit(1)

        mode = CFG["mode"]
        log.info(f"Running in WEBCAM mode  (mode={mode})")
        log.info("Frames are served via Flask — no local window.")

        while True:
            ret, frame = self._cap.read()
            if not ret:
                log.warning("cap.read() failed — retrying…")
                time.sleep(0.05)
                continue

            self._frame_count += 1

            # FPS calculation
            elapsed = time.time() - self._fps_timer
            if elapsed >= 1.0:
                self._fps         = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_timer   = time.time()

            # Frame skipping
            if self._frame_count % max(1, CFG["skip_frames"]) != 0:
                continue

            annotated, det = run_yolo(frame)
            draw_hud(annotated, det, "WEBCAM", self._fps, self._save_count)

            # ── WRITE to shared frame (Flask reads this) ──────────────────
            set_shared_frame(annotated)
            # ─────────────────────────────────────────────────────────────

            # Save logic
            should_save = (
                mode == "single"
                or mode == "continuous"
                or (mode == "on_detect" and det.count > 0)
            )

            if should_save:
                paths = self._saver.save(frame, annotated, det)
                self._save_count += 1
                log.info(
                    f"[{mode.upper()}] Saved #{self._save_count} | "
                    f"{det} → {list(paths.keys())}"
                )

            if mode == "single":
                log.info("Single capture done — exiting.")
                break

        self._cap.release()


# ─────────────────────────────────────────────────────────────────────────────
# ROS CAPTURE NODE  (writes to shared_frame for Flask)
# ─────────────────────────────────────────────────────────────────────────────
class CaptureNode(Node):
    def __init__(self):
        super().__init__("capture_node")
        self._log    = logging.getLogger("capture.node")
        self._bridge = CvBridge()
        self._saver  = FrameSaver(CFG["save_dir"])
        self._done   = threading.Event()

        self._frame_count  = 0
        self._save_count   = 0
        self._fps_timer    = time.time()
        self._fps          = 0.0
        self._last_frame_t = time.time()

        # ROS parameters (can be overridden at launch)
        self.declare_parameter("topic",       CFG["topic"])
        self.declare_parameter("conf",        CFG["conf"])
        self.declare_parameter("mode",        CFG["mode"])
        self.declare_parameter("skip_frames", CFG["skip_frames"])

        self._topic = self.get_parameter("topic").value       or CFG["topic"]
        self._conf  = self.get_parameter("conf").value        or CFG["conf"]
        self._mode  = self.get_parameter("mode").value        or CFG["mode"]
        self._skip  = self.get_parameter("skip_frames").value or CFG["skip_frames"]

        CFG["conf"] = self._conf   # keep global in sync

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=CFG["queue_depth"],
        )

        self._sub = self.create_subscription(
            Image, self._topic, self._callback, qos
        )
        self.create_timer(3.0, self._heartbeat)

        self._log.info(
            f"CaptureNode ready  topic='{self._topic}'  "
            f"mode='{self._mode}'  conf={self._conf}"
        )

    # ── heartbeat (warns if camera goes silent) ───────────────────────────
    def _heartbeat(self) -> None:
        gap = time.time() - self._last_frame_t
        if gap > 3.0:
            self._log.warning(f"No frame for {gap:.1f}s — check '{self._topic}'")

    # ── image callback ────────────────────────────────────────────────────
    def _callback(self, msg: "Image") -> None:
        if self._done.is_set():
            return

        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self._log.error(f"CvBridge: {exc}")
            return

        self._frame_count  += 1
        self._last_frame_t  = time.time()

        # FPS calculation
        elapsed = time.time() - self._fps_timer
        if elapsed >= 1.0:
            self._fps         = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_timer   = time.time()

        # Frame skipping
        if self._frame_count % max(1, self._skip) != 0:
            return

        annotated, det = run_yolo(frame)
        draw_hud(annotated, det, "ROS2", self._fps, self._save_count)

        # ── WRITE to shared frame (Flask reads this) ──────────────────────
        set_shared_frame(annotated)
        # ─────────────────────────────────────────────────────────────────

        # cv2.imshow(...) intentionally removed — Flask is the display layer

        # Save logic
        should_save = (
            self._mode == "single"
            or self._mode == "continuous"
            or (self._mode == "on_detect" and det.count > 0)
        )

        if should_save:
            paths = self._saver.save(frame, annotated, det)
            self._save_count += 1
            self._log.info(f"[{self._mode.upper()}] #{self._save_count} | {det}")

        if self._mode == "single":
            self._log.info("Single capture done.")
            self._shutdown()

    # ── clean shutdown ────────────────────────────────────────────────────
    def _shutdown(self) -> None:
        self._done.set()

    def destroy_node(self) -> None:
        super().destroy_node()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main(args=None) -> None:
    log.info("=" * 55)
    log.info(f"  Mode      : {CFG['mode']}")
    log.info(f"  YOLO      : {CFG['model']}  conf={CFG['conf']}")
    log.info(f"  Save dir  : {CFG['save_dir'].resolve()}")
    log.info(f"  Display   : Flask web stream (no local window)")
    log.info(f"  ROS 2     : {'available' if ROS_AVAILABLE else 'NOT found — webcam fallback'}")
    log.info("=" * 55)

    if not ROS_AVAILABLE:
        WebcamCapture().run()
        return

    # ── ROS 2 path ──────────────────────────────────────────────────────
    rclpy.init(args=args)
    node     = CaptureNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        log.info("Interrupted.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        sys.exit(0)


if __name__ == "__main__":
    main()
