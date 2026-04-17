import rclpy  # type: ignore[import]
from rclpy.node import Node  # type: ignore[import]
from sensor_msgs.msg import Image  # type: ignore[import]
from geometry_msgs.msg import Twist  # type: ignore[import]
from cv_bridge import CvBridge  # type: ignore[import]
import threading


class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')

        self.bridge = CvBridge()

        # Shared frame + lock (VERY IMPORTANT)
        self.frame = None
        self.frame_lock = threading.Lock()

        # Camera subscriber
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Velocity publisher (for movement)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.get_logger().info("Camera Subscriber Node Started 🚀")

    # ─────────────────────────────────────────────
    # CAMERA CALLBACK
    # ─────────────────────────────────────────────
    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Thread-safe write
            with self.frame_lock:
                self.frame = frame

        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")

    # ─────────────────────────────────────────────
    # GET FRAME (for Flask)
    # ─────────────────────────────────────────────
    def get_frame(self):
        with self.frame_lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    # ─────────────────────────────────────────────
    # SEND VELOCITY COMMAND
    # ─────────────────────────────────────────────
    def send_cmd_vel(self, linear=0.0, angular=0.0):
        twist = Twist()
        twist.linear.x = float(linear)
        twist.angular.z = float(angular)

        self.cmd_pub.publish(twist)


# ─────────────────────────────────────────────
# MAIN FUNCTION (SAFE SHUTDOWN)
# ─────────────────────────────────────────────

def main():
    rclpy.init()
    node = CameraSubscriber()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()