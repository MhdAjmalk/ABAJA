#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from geometry_msgs.msg import Twist

class BehaviorPlannerNode(Node):
    def __init__(self):
        super().__init__('behavior_planner')

        # FSM States
        self.state = 'WAIT_FOR_GREEN'

        # Parameters
        self.cruise_speed = 30.0  # kmph target cruise speed
        self.current_speed = 0.0  # vehicle current speed kmph
        self.speed_limit = None   # speed limit kmph, None if unknown
        self.lead_detected = False
        self.lead_distance = float('inf')  # meters
        self.lead_speed = 0.0               # kmph
        self.pedestrian_alert = False
        self.traffic_light = 'RED'          # RED, GREEN, YELLOW

        # Publishers: commands to ACC, AEB, LKA nodes
        self.acc_cmd_pub = self.create_publisher(Float32, '/acc/target_speed', 10)
        self.aeb_enable_pub = self.create_publisher(Bool, '/aeb/enable', 10)
        self.acc_enable_pub = self.create_publisher(Bool, '/acc/enable', 10)
        self.lka_enable_pub = self.create_publisher(Bool, '/lka/enable', 10)

        # Subscriptions: from perception and state topics
        self.create_subscription(String, '/traffic_light_state', self.traffic_light_cb, 10)
        self.create_subscription(Bool, '/pedestrian_alert', self.pedestrian_alert_cb, 10)
        self.create_subscription(Float32, '/vehicle_speed', self.vehicle_speed_cb, 10)
        self.create_subscription(String, '/lead_vehicle_info', self.lead_vehicle_cb, 10)  # custom string: "detected,distance,speed"
        self.create_subscription(Float32, '/speed_limit', self.speed_limit_cb, 10)

        # Timer for main loop
        self.timer = self.create_timer(0.1, self.fsm_step)  # 10Hz

        self.get_logger().info('Behavior Planner Node started')

    # Callbacks
    def traffic_light_cb(self, msg):
        self.traffic_light = msg.data.upper()

    def pedestrian_alert_cb(self, msg):
        self.pedestrian_alert = msg.data

    def vehicle_speed_cb(self, msg):
        self.current_speed = msg.data  # kmph

    def lead_vehicle_cb(self, msg):
        # Expecting format: "True,15.0,25.0"
        try:
            detected_str, dist_str, speed_str = msg.data.split(',')
            self.lead_detected = detected_str.lower() == 'true'
            self.lead_distance = float(dist_str)
            self.lead_speed = float(speed_str)
        except Exception as e:
            self.get_logger().warn(f'Failed to parse lead vehicle info: {e}')
            self.lead_detected = False
            self.lead_distance = float('inf')
            self.lead_speed = 0.0

    def speed_limit_cb(self, msg):
        # 0 means no limit known
        self.speed_limit = msg.data if msg.data > 0 else None

    # FSM safe following distance (meters)
    def safe_follow_distance(self):
        return max(8.0, self.current_speed * 1.0)  # 1s time gap or min 8m

    # Main FSM loop
    def fsm_step(self):
        # Priority: Emergency brake overrides all
        if self.pedestrian_alert:
            self.state = 'EMERGENCY_BRAKE'
            self.target_speed = 0.0
            self.enable_aeb(True)
            self.enable_acc(False)
            self.enable_lka(True)
            self.publish_target_speed(0.0)
            self.get_logger().info('State: EMERGENCY_BRAKE')
            return

        # State transitions
        if self.state == 'WAIT_FOR_GREEN':
            self.enable_aeb(False)
            self.enable_acc(False)
            self.enable_lka(False)
            self.publish_target_speed(0.0)
            if self.traffic_light == 'GREEN':
                self.state = 'ACCELERATE'
                self.target_speed = self.cruise_speed
                self.enable_acc(True)
                self.enable_lka(True)
                self.get_logger().info('State transition: WAIT_FOR_GREEN -> ACCELERATE')

        elif self.state == 'ACCELERATE':
            if self.current_speed >= self.target_speed - 1.0:
                self.state = 'CRUISE'
                self.get_logger().info('State transition: ACCELERATE -> CRUISE')

        elif self.state == 'CRUISE':
            if self.lead_detected and self.lead_distance < self.safe_follow_distance():
                self.state = 'FOLLOW_LEAD'
                self.get_logger().info('State transition: CRUISE -> FOLLOW_LEAD')

        elif self.state == 'FOLLOW_LEAD':
            if not self.lead_detected or self.lead_distance > self.safe_follow_distance():
                self.state = 'CRUISE'
                self.get_logger().info('State transition: FOLLOW_LEAD -> CRUISE')

        # Handle speed limits
        if self.speed_limit is not None:
            if self.state != 'SPEED_LIMIT':
                self.prev_state = self.state
                self.state = 'SPEED_LIMIT'
                self.get_logger().info(f'State transition: {self.prev_state} -> SPEED_LIMIT')
            self.target_speed = min(self.cruise_speed, self.speed_limit)
        else:
            if self.state == 'SPEED_LIMIT':
                self.state = getattr(self, 'prev_state', 'CRUISE')
                self.target_speed = self.cruise_speed
                self.get_logger().info(f'State transition: SPEED_LIMIT -> {self.state}')

        # Stop at red light if slow
        if self.traffic_light == 'RED' and self.current_speed < 1.0:
            self.state = 'WAIT_FOR_GREEN'
            self.target_speed = 0.0
            self.enable_acc(False)
            self.enable_aeb(False)
            self.enable_lka(False)
            self.get_logger().info('State transition: Any -> WAIT_FOR_GREEN')

        # Publish commands for ACC
        if self.state in ['CRUISE', 'ACCELERATE', 'FOLLOW_LEAD', 'SPEED_LIMIT']:
            self.enable_acc(True)
            self.enable_aeb(False)
            self.enable_lka(True)
            self.publish_target_speed(self.target_speed)

    # Helper functions to publish commands
    def publish_target_speed(self, speed):
        msg = Float32()
        msg.data = speed
        self.acc_cmd_pub.publish(msg)

    def enable_acc(self, enable):
        msg = Bool()
        msg.data = enable
        self.acc_enable_pub.publish(msg)

    def enable_aeb(self, enable):
        msg = Bool()
        msg.data = enable
        self.aeb_enable_pub.publish(msg)

    def enable_lka(self, enable):
        msg = Bool()
        msg.data = enable
        self.lka_enable_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = BehaviorPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Behavior Planner shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

