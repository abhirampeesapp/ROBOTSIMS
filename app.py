import os
import sys
import time
import threading
import logging
import atexit
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, render_template
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rover")

# ─────────────────────────────────────────────────────────────────────────────
# YOLO  —  load once, warm up, keep global
# ─────────────────────────────────────────────────────────────────────────────
YOLO_MODEL_PATH  = os.environ.get("YOLO_MODEL", "yolov8n.pt")
YOLO_CONF        = float(os.environ.get("YOLO_CONF", "0.5"))
YOLO_SKIP_FRAMES = int(os.environ.get("YOLO_SKIP", "2"))

log.info(f"Loading YOLO model: {YOLO_MODEL_PATH}")
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    _ = yolo_model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
    log.info("YOLO model ready ✓")
except Exception as exc:
    log.critical(f"Failed to load YOLO model: {exc}")
    sys.exit(1)

DETECTION_COLORS = {
    "person": (0, 255, 0),
    "snake":  (0, 0, 255),
    "fire":   (0, 80, 255),
    "car":    (255, 165, 0),
    "truck":  (255, 100, 0),
    "bus":    (255, 50,  0),
}
DEFAULT_COLOR = (0, 165, 255)

# ─────────────────────────────────────────────────────────────────────────────
# ROS 2  —  guarded import
# ─────────────────────────────────────────────────────────────────────────────
try:
    import rclpy                                                    # type: ignore[import]
    from rclpy.node import Node                                     # type: ignore[import]
    from rclpy.executors import MultiThreadedExecutor               # type: ignore[import]
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy  # type: ignore[import]
    from sensor_msgs.msg import Image                               # type: ignore[import]
    from geometry_msgs.msg import Twist                             # type: ignore[import]
    from cv_bridge import CvBridge, CvBridgeError                   # type: ignore[import]
    ROS_AVAILABLE = True
    log.info("ROS 2 packages found ✓")
except ImportError:
    ROS_AVAILABLE = False
    log.warning("ROS 2 not found — running in webcam/placeholder mode")

# ─────────────────────────────────────────────────────────────────────────────
# RECORDING
# ─────────────────────────────────────────────────────────────────────────────
SAVE_PATH         = os.environ.get("SAVE_PATH", "recordings")
RECORDING_ENABLED = os.environ.get("RECORDING", "1") == "1"
MIN_CROP_PIXELS   = 1600

os.makedirs(SAVE_PATH, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SHARED STATE
# ─────────────────────────────────────────────────────────────────────────────
start_time        = time.time()
latest_detections = []
detections_lock   = threading.Lock()

telemetry = {
    "battery":    100,
    "signal":     "Strong",
    "uptime":     0,
    "fps":        0,
    "mode":       "ros2" if ROS_AVAILABLE else "simulation",
    "detections": 0,
}

# ─────────────────────────────────────────────────────────────────────────────
# SHARED FRAME  ← replaces internal camera reads in gen_frames()
# capture.py writes here; this file reads from here.
# ─────────────────────────────────────────────────────────────────────────────
_shared_frame: np.ndarray | None = None
_shared_frame_lock = threading.Lock()


def set_shared_frame(frame: np.ndarray) -> None:
    """Called by the capture backend (ROS node or webcam loop) to push a frame."""
    global _shared_frame
    with _shared_frame_lock:
        _shared_frame = frame.copy()


def get_shared_frame() -> np.ndarray | None:
    """Called by gen_frames() to pull the latest frame."""
    with _shared_frame_lock:
        return _shared_frame.copy() if _shared_frame is not None else None


# ─────────────────────────────────────────────────────────────────────────────
# PLACEHOLDER FRAME  (shown before the first real frame arrives)
# ─────────────────────────────────────────────────────────────────────────────
def _placeholder_frame() -> np.ndarray:
    h, w  = 480, 640
    frame = np.full((h, w, 3), (8, 18, 24), dtype=np.uint8)
    for x in range(0, w, 40):
        cv2.line(frame, (x, 0), (x, h), (0, 40, 50), 1)
    for y in range(0, h, 40):
        cv2.line(frame, (0, y), (w, y), (0, 40, 50), 1)
    cv2.putText(frame, "NO CAMERA SIGNAL",
                (140, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 180, 200), 2)
    cv2.putText(frame, "Connect camera or enable ROS 2",
                (100, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 140), 1)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# ROS 2 CAMERA NODE
# ─────────────────────────────────────────────────────────────────────────────
if ROS_AVAILABLE:
    class ROSCamera(Node):
        """
        Subscribes to /camera/image_raw.
        Writes raw frames directly to set_shared_frame() — no imshow.
        Publishes Twist to /cmd_vel via send_cmd_vel().
        Spins on its own daemon thread; Flask never blocks on ROS.
        """

        def __init__(self):
            super().__init__("flask_camera_node")
            self._logger = logging.getLogger("rover.ros")
            self._bridge = CvBridge()
            self._last_frame_time = time.time()

            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
            )

            self._sub = self.create_subscription(
                Image, "/camera/image_raw", self._image_callback, qos
            )
            self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
            self.create_timer(3.0, self._heartbeat)
            self._logger.info("ROSCamera node initialised ✓")

        def _image_callback(self, msg: Image) -> None:
            try:
                frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except CvBridgeError as exc:
                self._logger.error(f"CvBridge error: {exc}")
                return

            # ── push raw frame to shared buffer; gen_frames() reads it ──
            set_shared_frame(frame)
            self._last_frame_time = time.time()

        def _heartbeat(self) -> None:
            gap = time.time() - self._last_frame_time
            if gap > 3.0:
                self._logger.warning(
                    f"No camera frame for {gap:.1f}s — "
                    "check /camera/image_raw"
                )

        def send_cmd_vel(self, linear: float, angular: float) -> bool:
            try:
                twist = Twist()
                twist.linear.x  = float(linear)
                twist.angular.z = float(angular)
                self._cmd_pub.publish(twist)
                return True
            except Exception as exc:
                self._logger.error(f"cmd_vel publish failed: {exc}")
                return False


# ─────────────────────────────────────────────────────────────────────────────
# WEBCAM FALLBACK
# ─────────────────────────────────────────────────────────────────────────────
class WebcamCamera:
    """
    Opens the first available webcam and pushes frames to set_shared_frame().
    Runs a background capture loop — Flask never calls cap.read() directly.
    Falls back to placeholder if no webcam is found.
    """

    def __init__(self):
        self._cap = self._open_camera()
        if self._cap:
            threading.Thread(
                target=self._capture_loop,
                daemon=True,
                name="webcam-capture",
            ).start()
        else:
            log.warning("No webcam found — serving placeholder frames")

    def _open_camera(self) -> cv2.VideoCapture | None:
        backends = [(cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_ANY, "default")]
        for idx in range(3):
            for backend, bname in backends:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        log.info(f"Webcam opened: index={idx} backend={bname} ✓")
                        return cap
                cap.release()
        return None

    def _capture_loop(self) -> None:
        """Continuously reads from the webcam and pushes to shared frame."""
        while self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret:
                set_shared_frame(frame)
            else:
                log.warning("cap.read() returned False — retrying…")
                time.sleep(0.1)

    # cmd_vel is a no-op in webcam mode (no robot to drive)
    def send_cmd_vel(self, linear: float, angular: float) -> bool:
        log.debug(f"[webcam stub] cmd_vel linear={linear} angular={angular}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# INITIALISE CAMERA BACKEND
# ─────────────────────────────────────────────────────────────────────────────
if ROS_AVAILABLE:
    rclpy.init()
    camera_node = ROSCamera()

    _executor = MultiThreadedExecutor()
    _executor.add_node(camera_node)

    _ros_thread = threading.Thread(
        target=_executor.spin,
        daemon=True,
        name="ros2-executor",
    )
    _ros_thread.start()
    log.info("ROS 2 executor spinning on background thread ✓")

else:
    camera_node = WebcamCamera()

# ─────────────────────────────────────────────────────────────────────────────
# YOLO PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
_yolo_lock      = threading.Lock()
_last_annotated: np.ndarray | None = None


def process_with_yolo(frame: np.ndarray) -> np.ndarray:
    """Run YOLOv8 on *frame*; update detections; optionally save crops."""
    global _last_annotated

    with _yolo_lock:
        try:
            results = yolo_model(frame, conf=YOLO_CONF, verbose=False)
        except Exception as exc:
            log.error(f"YOLO inference error: {exc}")
            return frame

    annotated    = frame.copy()
    frame_labels = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf   = float(box.conf[0])
            cls_id = int(box.cls[0])
            label  = r.names[cls_id]

            frame_labels.append({"label": label, "conf": round(conf, 3)})

            color = DETECTION_COLORS.get(label, DEFAULT_COLOR)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                f"{label} {conf:.2f}",
                (x1, max(y1 - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )

            if RECORDING_ENABLED:
                crop = frame[y1:y2, x1:x2]
                if crop.size >= MIN_CROP_PIXELS:
                    ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    out_path = os.path.join(SAVE_PATH, f"{label}_{ts}_crop.jpg")
                    cv2.imwrite(out_path, crop)

    with detections_lock:
        latest_detections[:] = frame_labels

    telemetry["detections"] = len(frame_labels)
    _last_annotated = annotated
    return annotated


# ─────────────────────────────────────────────────────────────────────────────
# FRAME GENERATOR  ← now reads from get_shared_frame() instead of camera_node
# ─────────────────────────────────────────────────────────────────────────────
def gen_frames(yolo: bool = False):
    frame_counter = 0
    fps_counter   = 0
    fps_timer     = time.time()

    while True:
        t0 = time.time()

        # ── read from the shared buffer written by ROS / webcam loop ──────
        frame = get_shared_frame()

        if frame is None:
            # No frame yet — show placeholder until camera is live
            frame = _placeholder_frame()
        elif yolo:
            frame_counter += 1
            if frame_counter % YOLO_SKIP_FRAMES == 0:
                # Full inference every Nth frame
                frame = process_with_yolo(frame)
            elif _last_annotated is not None:
                # Reuse last annotated output between inference frames
                frame = _last_annotated.copy()

        # FPS + uptime telemetry (updated every second)
        fps_counter += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            telemetry["fps"]    = round(fps_counter / elapsed, 1)
            telemetry["uptime"] = round(time.time() - start_time)
            fps_counter = 0
            fps_timer   = time.time()

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            time.sleep(0.033)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buf.tobytes()
            + b"\r\n"
        )

        # Cap at ~30 FPS
        time.sleep(max(0.0, 0.033 - (time.time() - t0)))


# ─────────────────────────────────────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/")
def dashboard():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(yolo=False),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/video_feed_yolo")
def video_feed_yolo():
    return Response(
        gen_frames(yolo=True),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/detections")
def get_detections():
    with detections_lock:
        snapshot = list(latest_detections)
    return jsonify({"objects": snapshot, "count": len(snapshot)})


@app.route("/telemetry")
def get_telemetry():
    telemetry["uptime"] = round(time.time() - start_time)
    return jsonify(telemetry)


@app.route("/cmd_vel", methods=["POST"])
def cmd_vel():
    data    = request.get_json(silent=True) or {}
    linear  = float(data.get("linear",  0.0))
    angular = float(data.get("angular", 0.0))

    # Clamp to safe velocity range
    linear  = max(-2.0, min(2.0, linear))
    angular = max(-2.0, min(2.0, angular))

    ok = camera_node.send_cmd_vel(linear, angular)
    return jsonify({"success": ok, "linear": linear, "angular": angular})


@app.route("/status")
def status():
    cam_ok = get_shared_frame() is not None   # reflects actual frame presence
    return jsonify({
        "ros_available": ROS_AVAILABLE,
        "camera_live":   cam_ok,
        "yolo_model":    YOLO_MODEL_PATH,
        "yolo_conf":     YOLO_CONF,
        "yolo_skip":     YOLO_SKIP_FRAMES,
        "recording":     RECORDING_ENABLED,
        "save_path":     SAVE_PATH,
        "uptime":        round(time.time() - start_time),
    })


@app.route("/config", methods=["POST"])
def update_config():
    """Live-update YOLO confidence and skip-frames without restart."""
    global YOLO_CONF, YOLO_SKIP_FRAMES
    data = request.get_json(silent=True) or {}
    if "conf" in data:
        YOLO_CONF = float(max(0.05, min(1.0, data["conf"])))
    if "skip" in data:
        YOLO_SKIP_FRAMES = int(max(1, data["skip"]))
    return jsonify({"conf": YOLO_CONF, "skip": YOLO_SKIP_FRAMES})


# ─────────────────────────────────────────────────────────────────────────────
# GRACEFUL SHUTDOWN
# ─────────────────────────────────────────────────────────────────────────────
def _shutdown():
    log.info("Shutting down…")
    if ROS_AVAILABLE:
        try:
            _executor.shutdown()
            camera_node.destroy_node()
            rclpy.shutdown()
            log.info("ROS 2 shut down cleanly ✓")
        except Exception as exc:
            log.warning(f"ROS shutdown error (non-fatal): {exc}")

atexit.register(_shutdown)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=" * 60)
    log.info("ROS 2 Rover Dashboard")
    log.info(f"  YOLO model   : {YOLO_MODEL_PATH}  (conf={YOLO_CONF}, skip={YOLO_SKIP_FRAMES})")
    log.info(f"  Recording    : {'ENABLED → ' + SAVE_PATH if RECORDING_ENABLED else 'DISABLED'}")
    log.info(f"  ROS 2        : {'ENABLED' if ROS_AVAILABLE else 'DISABLED (webcam/placeholder)'}")
    log.info("  Dashboard    → http://127.0.0.1:5000")
    log.info("  Raw feed     → http://127.0.0.1:5000/video_feed")
    log.info("  YOLO feed    → http://127.0.0.1:5000/video_feed_yolo")
    log.info("  Detections   → http://127.0.0.1:5000/detections")
    log.info("  Telemetry    → http://127.0.0.1:5000/telemetry")
    log.info("  Status       → http://127.0.0.1:5000/status")
    log.info("  Config POST  → http://127.0.0.1:5000/config")
    log.info("=" * 60)

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)