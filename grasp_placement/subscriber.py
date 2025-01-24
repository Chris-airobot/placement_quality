import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.qos import QoSProfile
from rclpy.wait_for_message import wait_for_message
from tf2_msgs.msg import TFMessage

class TFSubscriber(Node):
    def __init__(self):
        super().__init__("tf_subscriber")
        self.latest_tf = None  # Store the latest TFMessage here

        # Create the subscription
        self.subscription = self.create_subscription(
            TFMessage,           # Message type
            "tf",               # Topic name
            self.tf_callback,    # Callback function
            10                   # QoS
        )

    def tf_callback(self, msg):
        # This callback is triggered for every new TF message on /tf
        self.latest_tf = msg

def main():
    rclpy.init()
    
    # Create a node
    node = TFSubscriber()

    
    rclpy.spin_once(node)

    print(node.latest_tf)
    # Clean up and shut down
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
