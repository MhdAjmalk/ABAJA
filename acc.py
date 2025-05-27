#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from time import time
from rcl_interfaces.msg import SetParametersResult
from collections import deque

# Import required message types
from std_msgs.msg import Float64, Bool
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
from radar_msgs.msg import RadarScan
import json

class HybridACCNode(Node):
    def __init__(self):
        super().__init__('hybrid_acc_node')
        
        # Declare parameters
        self.declare_parameter('min_distance', 5.0)     # Minimum safety distance (m)
        self.declare_parameter('target_distance', 8.0)  # Target following distance (m)
        self.declare_parameter('time_gap', 1.2)         # Time gap for CTG algorithm (s)
        self.declare_parameter('max_speed', 10.0)       # Maximum speed (m/s)
        self.declare_parameter('reaction_time', 1.0)    # Vehicle reaction time (s)
        self.declare_parameter('pid_kp', 0.4)           # PID proportional gain
        self.declare_parameter('pid_ki', 0.05)          # PID integral gain
        self.declare_parameter('pid_kd', 0.2)           # PID derivative gain
        self.declare_parameter('throttle_limit', 1.0)   # Max throttle (0-1)
        self.declare_parameter('brake_limit', 1.0)      # Max brake (0-1)
        self.declare_parameter('jerk_limit', 0.5)       # Max jerk limit (m/s³)
        self.declare_parameter('enable_acc', True)      # ACC enabled by default
        
        # Get parameters
        self.min_distance = self.get_parameter('min_distance').value
        self.target_distance = self.get_parameter('target_distance').value
        self.time_gap = self.get_parameter('time_gap').value
        self.max_speed = self.get_parameter('max_speed').value
        self.reaction_time = self.get_parameter('reaction_time').value
        self.pid_kp = self.get_parameter('pid_kp').value
        self.pid_ki = self.get_parameter('pid_ki').value
        self.pid_kd = self.get_parameter('pid_kd').value
        self.throttle_limit = self.get_parameter('throttle_limit').value
        self.brake_limit = self.get_parameter('brake_limit').value
        self.jerk_limit = self.get_parameter('jerk_limit').value
        self.acc_enabled = self.get_parameter('enable_acc').value
        
        # Setup parameter callback
        self.add_on_set_parameters_callback(self.parameters_callback)
        
        # Vehicle state
        self.current_speed = 0.0            # Current vehicle speed (m/s)
        self.lead_distance = float('inf')   # Distance to lead vehicle (m)
        self.lead_speed = 0.0               # Estimated speed of lead vehicle (m/s)
        self.lead_detected = False          # Flag if lead vehicle is detected
        self.acceleration = 0.0             # Current vehicle acceleration (m/s²)
        self.last_acceleration_time = time()
        
        # PID controller state
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = time()
        
        # Reaction time compensation (sliding window for commands)
        self.control_buffer = deque(maxlen=int(20 * self.reaction_time))  # Buffer for 20Hz x reaction_time
        self.last_throttle = 0.0
        self.last_brake = 0.0
        
        # IMU data
        self.linear_accel_x = 0.0
        self.imu_timestamp = 0.0
        self.using_imu_accel = False
        
        # System status
        self.acc_ready = False              # ACC ready to engage
        
        # Debug/monitoring variables
        self.control_value = 0.0
        self.target_speed = 0.0
        self.computed_target_distance = 0.0
        self.distance_error = 0.0
        
        # Create subscribers
        self.create_subscription(
            Float64,
            '/wheelspeed',
            self.speed_callback,
            10)
        
        self.create_subscription(
            RadarScan,
            '/radar',
            self.radar_callback,
            10)
        
        self.create_subscription(
            Imu,
            '/imu',
            self.imu_callback,
            10)
        
        self.create_subscription(
            Bool,
            '/acc_enable',
            self.enable_callback,
            10)
        
        # Create publishers
        # Vehicle control publisher (only controlling y=acceleration, z=braking)
        self.control_publisher = self.create_publisher(
            Twist,
            '/vehicleControl',
            10)
        
        # Status publisher
        self.status_publisher = self.create_publisher(
            Twist,
            '/acc_status',
            10)
        
        # Control loop timer (20 Hz)
        self.timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info('Hybrid ACC Node initialized with parameters:')
        self.get_logger().info(f'  Min distance: {self.min_distance}m')
        self.get_logger().info(f'  Target distance: {self.target_distance}m')
        self.get_logger().info(f'  Time gap: {self.time_gap}s')
        self.get_logger().info(f'  Max speed: {self.max_speed}m/s')
        self.get_logger().info(f'  Reaction time: {self.reaction_time}s')
        self.get_logger().info(f'  PID gains: kp={self.pid_kp}, ki={self.pid_ki}, kd={self.pid_kd}')
        self.get_logger().info(f'  ACC enabled: {self.acc_enabled}')
    
    def parameters_callback(self, params):
        """Parameter update callback"""
        result = SetParametersResult()
        result.successful = True
        
        for param in params:
            self.get_logger().info(f'Parameter {param.name} changed to {param.value}')
            if param.name == 'min_distance':
                self.min_distance = param.value
            elif param.name == 'target_distance':
                self.target_distance = param.value
            elif param.name == 'time_gap':
                self.time_gap = param.value
            elif param.name == 'max_speed':
                self.max_speed = param.value
            elif param.name == 'reaction_time':
                self.reaction_time = param.value
                # Update buffer size
                self.control_buffer = deque(maxlen=int(20 * self.reaction_time))
            elif param.name == 'pid_kp':
                self.pid_kp = param.value
            elif param.name == 'pid_ki':
                self.pid_ki = param.value
            elif param.name == 'pid_kd':
                self.pid_kd = param.value
            elif param.name == 'throttle_limit':
                self.throttle_limit = param.value
            elif param.name == 'brake_limit':
                self.brake_limit = param.value
            elif param.name == 'jerk_limit':
                self.jerk_limit = param.value
            elif param.name == 'enable_acc':
                old_state = self.acc_enabled
                self.acc_enabled = param.value
                if old_state and not self.acc_enabled:
                    self.reset_controller()
        
        return result
    
    def reset_controller(self):
        """Reset controller state"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.control_buffer.clear()
        self.last_throttle = 0.0
        self.last_brake = 0.0
        self.publish_control(0.0, 0.0)
    
    def speed_callback(self, msg):
        """Wheel speed callback"""
        self.current_speed = msg.data  # m/s
        
        # Estimate acceleration if IMU data is not available
        if not self.using_imu_accel:
            current_time = time()
            dt = current_time - self.last_acceleration_time
            if dt > 0.02:  # Only update if enough time has passed (50Hz)
                # Simple acceleration estimate using speed difference
                if hasattr(self, 'last_speed'):
                    self.acceleration = (self.current_speed - self.last_speed) / dt
                self.last_speed = self.current_speed
                self.last_acceleration_time = current_time
        
        # Update ACC ready status
        if not self.acc_ready and self.current_speed > 0.5:  # Only enable when moving
            self.acc_ready = True
    
    def radar_callback(self, msg):
        """Radar data callback"""
        # Find the closest relevant object in front
        min_distance = float('inf')
        lead_speed_rel = 0.0
        lead_accel = 0.0
        
        for track in msg.tracks:
            # Basic filtering - objects in our path
            # Assuming radar coordinates: x=forward, y=lateral
            if (abs(track.position.y) < 2.0 and 
                track.position.x > 0 and 
                track.position.x < 50.0):  # Within 2m laterally, in front, < 50m
                
                # Get object distance
                distance = track.position.x
                
                # Filter for objects in our lane/path
                if distance < min_distance:
                    min_distance = distance
                    lead_speed_rel = track.velocity.x  # Relative velocity in x direction
                    lead_accel = track.acceleration.x if hasattr(track, 'acceleration') else 0.0
        
        if min_distance < 50.0:  # Valid detection within 50m
            self.lead_detected = True
            self.lead_distance = min_distance
            self.lead_speed = self.current_speed + lead_speed_rel  # Absolute speed
            
            # Safety check - lead speed shouldn't be negative
            if self.lead_speed < 0:
                self.lead_speed = 0.0
        else:
            self.lead_detected = False
            self.lead_distance = float('inf')
    
    def imu_callback(self, msg):
        """IMU data callback to get acceleration data"""
        # Extract linear acceleration in x direction (forward)
        self.linear_accel_x = msg.linear_acceleration.x
        self.imu_timestamp = time()
        self.using_imu_accel = True
        
        # Update current acceleration (filtered)
        # Low-pass filter to reduce noise (alpha = 0.3)
        self.acceleration = 0.3 * self.linear_accel_x + 0.7 * self.acceleration
    
    def enable_callback(self, msg):
        """ACC enable/disable callback"""
        old_state = self.acc_enabled
        self.acc_enabled = msg.data
        
        # Reset controller when disabling
        if old_state and not self.acc_enabled:
            self.reset_controller()
    
    def compute_target_distance(self):
        """Enhanced Constant Time-Gap (CTG) algorithm with reaction time compensation"""
        # Dynamic distance based on speed (CTG) + minimum fixed distance
        # Add reaction time buffer for enhanced safety
        dynamic_distance = (self.min_distance + 
                           (self.time_gap * self.current_speed) + 
                           (self.reaction_time * self.current_speed * 0.5))
        
        # Use the higher of target distance or dynamic distance
        return max(self.target_distance, dynamic_distance)
    
    def compute_predictive_distance(self, dt=1.0):
        """Predict future distance to lead vehicle based on current speeds and accelerations"""
        if not self.lead_detected:
            return float('inf')
        
        # Predict future positions using kinematic equations
        future_ego_position = self.current_speed * dt + 0.5 * self.acceleration * dt * dt
        future_lead_position = self.lead_distance + self.lead_speed * dt
        
        # Calculate future distance
        future_distance = future_lead_position - future_ego_position
        
        return max(0.1, future_distance)  # Ensure positive value
    
    def compute_pid_control(self, error, dt):
        """PID control for acceleration/braking with enhanced derivative term"""
        if dt <= 0:
            return 0.0
            
        # P term
        p_term = self.pid_kp * error
        
        # I term with anti-windup
        self.integral += error * dt
        
        # Anti-windup measures
        if (error >= 0 and self.integral < 0) or (error <= 0 and self.integral > 0):
            # Reset integral when error changes sign (avoid oscillation)
            self.integral = 0.0
            
        self.integral = np.clip(self.integral, -10.0, 10.0)  # Limit integral windup
        i_term = self.pid_ki * self.integral
        
        # D term with filtering for noise reduction
        d_raw = (error - self.prev_error) / dt
        # Low-pass filter for derivative (alpha = 0.2)
        d_filtered = 0.2 * d_raw + 0.8 * (self.prev_error / dt if dt > 0 else 0)
        d_term = self.pid_kd * d_filtered
        
        # Update previous error
        self.prev_error = error
        
        # Combined control output with limits
        control = p_term + i_term + d_term
        
        return control
    
    def limit_jerk(self, new_control, old_control, dt):
        """Limit jerk (rate of change of acceleration) for comfort"""
        if dt <= 0:
            return new_control
            
        # Calculate current jerk
        control_change = new_control - old_control
        current_jerk = control_change / dt
        
        # Limit jerk
        if abs(current_jerk) > self.jerk_limit:
            max_change = self.jerk_limit * dt
            # Limit the change
            if control_change > max_change:
                return old_control + max_change
            elif control_change < -max_change:
                return old_control - max_change
                
        return new_control
    
    def publish_control(self, throttle, brake):
        """Publish vehicle control commands"""
        # Create control message
        control_msg = Twist()
        
        # Assign control values to appropriate fields
        # y=acceleration, z=braking as per your specification
        control_msg.linear.y = throttle       # Throttle command
        control_msg.linear.z = brake          # Brake command
        
        # Publish control command
        self.control_publisher.publish(control_msg)
        
        # Save last values
        self.last_throttle = throttle
        self.last_brake = brake
    
    def publish_status(self):
        """Publish ACC status information"""
        status_msg = Twist()
        
        # Pack status information into Twist message fields
        status_msg.linear.x = self.current_speed         # Current speed
        status_msg.linear.y = self.target_speed          # Target speed
        status_msg.linear.z = self.lead_distance         # Lead vehicle distance
        status_msg.angular.x = float(self.lead_detected) # Lead detected flag
        status_msg.angular.y = float(self.acc_enabled)   # ACC enabled flag
        status_msg.angular.z = self.computed_target_distance # Target following distance
        
        # Publish status
        self.status_publisher.publish(status_msg)
    
    def control_loop(self):
        """Main control loop for ACC"""
        # Skip processing if ACC is not enabled or not ready
        if not self.acc_enabled or not self.acc_ready:
            # Zero throttle/brake when not active
            self.publish_control(0.0, 0.0)
            return
        
        # Calculate time delta
        current_time = time()
        dt = current_time - self.prev_time
        self.prev_time = current_time
        
        if dt <= 0:
            return
        
        # Compute target distance using CTG with reaction time compensation
        self.computed_target_distance = self.compute_target_distance()
        
        # Determine target speed based on lead vehicle detection
        if self.lead_detected:
            # Look ahead to predict distance after reaction time
            predicted_distance = self.compute_predictive_distance(dt=self.reaction_time)
            
            # Calculate distance error (positive = too far, negative = too close)
            self.distance_error = predicted_distance - self.computed_target_distance
            
            # Adapt target speed based on distance error with predictive component
            if self.distance_error < 0:
                # Too close - need to slow down
                # Calculate deceleration based on closeness
                closeness_factor = min(1.0, abs(self.distance_error) / self.computed_target_distance)
                
                # More aggressive braking when very close
                if predicted_distance < self.min_distance:
                    # Emergency situation - rapid deceleration
                    self.target_speed = max(0.0, self.lead_speed * 0.5)
                else:
                    # Normal deceleration - proportional to how close we are
                    self.target_speed = max(0.0, self.lead_speed * (1.0 - closeness_factor * 0.3))
            else:
                # Far enough - can match or gradually approach lead vehicle speed
                # Faster catch-up when further away
                catch_up_factor = min(1.0, self.distance_error / (3.0 * self.computed_target_distance))
                self.target_speed = min(self.lead_speed + catch_up_factor * 2.0, self.max_speed)
        else:
            # No lead vehicle - maintain max allowed speed
            self.target_speed = self.max_speed
        
        # Calculate speed error for controller
        speed_error = self.target_speed - self.current_speed
        
        # Compute control signal using PID
        raw_control = self.compute_pid_control(speed_error, dt)
        
        # Apply jerk limiting for comfort
        limited_control = self.limit_jerk(raw_control, self.control_value, dt)
        self.control_value = limited_control
        
        # Convert control value to throttle and brake commands
        if self.control_value >= 0:
            # Positive control = accelerate
            throttle = min(self.control_value, self.throttle_limit)
            brake = 0.0
        else:
            # Negative control = brake
            throttle = 0.0
            brake = min(-self.control_value, self.brake_limit)
        
        # Store commands in buffer to compensate for reaction time
        self.control_buffer.append((throttle, brake))
        
        # Get command from buffer (delayed by reaction time) or use current if buffer not full
        if len(self.control_buffer) >= int(20 * self.reaction_time):
            # Use oldest command in buffer
            throttle_cmd, brake_cmd = self.control_buffer[0]
        else:
            # Buffer not full yet, use current command but with reduced values
            fill_factor = max(0.1, len(self.control_buffer) / (20 * self.reaction_time))
            throttle_cmd = throttle * fill_factor
            brake_cmd = brake * fill_factor
        
        # Apply throttle and brake commands
        self.publish_control(throttle_cmd, brake_cmd)
        
        # Publish status information
        self.publish_status()
        
        # Debug logging
        self.get_logger().debug(
            f"Speed: {self.current_speed:.1f}/{self.target_speed:.1f} m/s, "
            f"Lead: {self.lead_distance:.1f}m, Predict: {predicted_distance:.1f}m, "
            f"Accel: {self.acceleration:.2f}m/s², Control: {self.control_value:.2f}, "
            f"Buffer: {len(self.control_buffer)}/{int(20 * self.reaction_time)}, "
            f"Cmd: T={throttle_cmd:.2f}, B={brake_cmd:.2f}"
        )

def main(args=None):
    rclpy.init(args=args)
    acc_node = HybridACCNode()
    
    try:
        rclpy.spin(acc_node)
    except KeyboardInterrupt:
        acc_node.get_logger().info('Keyboard interrupt, shutting down ACC')
    finally:
        # Safety: ensure zero throttle and brake on shutdown
        acc_node.publish_control(0.0, 0.0)
        acc_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()