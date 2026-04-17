import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import termios
import tty

class Teleop(Node):
    def __init__(self):
        super().__init__('teleop_keyboard')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

    def move(self, lin, ang):
        msg = Twist()
        msg.linear.x = lin
        msg.angular.z = ang
        self.pub.publish(msg)


def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    return key


def main():
    rclpy.init()
    node = Teleop()

    print("""
    🔥 TELEOP CONTROLS

        w
    a   s   d

    w → forward
    s → stop
    a → left
    d → right
    x → back
    q → quit
    """)

    try:
        while True:
            key = get_key()

            if key == 'w':
                node.move(0.3, 0.0)
            elif key == 'x':
                node.move(-0.3, 0.0)
            elif key == 'a':
                node.move(0.0, 0.8)
            elif key == 'd':
                node.move(0.0, -0.8)
            elif key == 's':
                node.move(0.0, 0.0)
            elif key == 'q':
                break

            rclpy.spin_once(node)

    except KeyboardInterrupt:
        pass

    # Stop robot before exit
    node.move(0.0, 0.0)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
