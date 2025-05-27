#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from time import time
import matplotlib.pyplot as plt
import json
import math
from rcl_interfaces.msg import SetParametersResult

# Import required message types
from std_msgs.msg import String, Float64
from sensor_msgs.msg import NavSatFix, Imu
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path

class PIDController:
    def __init__(self, kp, ki, kd, integral_limit=10):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0
        self.integral_limit = integral_limit

    def compute(self, error, dt):
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0

class StanleyControllerNode(Node):
    def __init__(self):
        super().__init__('stanley_controller_node')
        
        # Controller parameters
        self.declare_parameter('k', 0.2)
        self.declare_parameter('pid_kp', 0.1)
        self.declare_parameter('pid_ki', 0.01)
        self.declare_parameter('pid_kd', 0.05)
        self.declare_parameter('lookahead_distance', 2.0)
        self.declare_parameter('smoothing_window', 5)
        
        # Get parameters
        self.k = self.get_parameter('k').value  # Stanley gain
        kp = self.get_parameter('pid_kp').value
        ki = self.get_parameter('pid_ki').value
        kd = self.get_parameter('pid_kd').value
        self.lookahead_distance = self.get_parameter('lookahead_distance').value
        self.smoothing_window = self.get_parameter('smoothing_window').value
        
        self.pid_controller = PIDController(kp=kp, ki=ki, kd=kd, integral_limit=5)
        
        # Setup parameter callback
        self.add_on_set_parameters_callback(self.parameters_callback)
        
        # Vehicle state
        self.vehicle_x = 0.0
        self.vehicle_y = 0.0
        self.vehicle_yaw = 0.0
        self.vehicle_speed = 0.0
        self.last_steering_angle = 0.0
        
        # Reference path
        self.raw_waypoints = []
        self.waypoints = []
        self.reference_path_received = False
        
        # Time tracking
        self.prev_time = time()
        self.start_time = time()
        
        # Data for logging and plotting
        self.time_list = []
        self.steering_angle_list = []
        self.cte_list = []
        
        # Create subscribers
        self.create_subscription(
            Imu,
            '/imu',
            self.imu_callback,
            10)
        
        self.create_subscription(
            NavSatFix,
            '/gnss',
            self.gnss_callback,
            10)
        
        self.create_subscription(
            Float64,
            '/wheelspeed',
            self.wheelspeed_callback,
            10)
        
        self.create_subscription(
            Float64,
            '/steeringangle',
            self.steering_angle_callback,
            10)
        
        # Subscribe to lane coordinates as JSON string
        self.create_subscription(
            String,
            '/lane_coordinates',
            self.lane_coordinates_callback,
            10)
        
        # Create publishers
        self.control_publisher = self.create_publisher(
            Twist,
            '/vehicleControl',
            10)
        
        # Publish processed path for visualization
        self.path_publisher = self.create_publisher(
            Path,
            '/processed_path',
            10)
        
        # Create timer for control loop
        self.timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info('Stanley Controller Node initialized with parameters:')
        self.get_logger().info(f'  Stanley gain (k): {self.k}')
        self.get_logger().info(f'  PID gains: kp={kp}, ki={ki}, kd={kd}')
        self.get_logger().info(f'  Lookahead distance: {self.lookahead_distance}m')
        self.get_logger().info(f'  Path smoothing window: {self.smoothing_window} points')

    def parameters_callback(self, params):
        result = SetParametersResult()
        result.successful = True
        
        for param in params:
            self.get_logger().info(f'Parameter {param.name} changed to {param.value}')
            if param.name == 'k':
                self.k = param.value
            elif param.name == 'pid_kp':
                self.pid_controller.kp = param.value
            elif param.name == 'pid_ki':
                self.pid_controller.ki = param.value
            elif param.name == 'pid_kd':
                self.pid_controller.kd = param.value
            elif param.name == 'lookahead_distance':
                self.lookahead_distance = param.value
            elif param.name == 'smoothing_window':
                self.smoothing_window = param.value
                # Re-smooth the path if we have waypoints
                if self.raw_waypoints:
                    self.waypoints = self.smooth_path(self.raw_waypoints, self.smoothing_window)
                    self.publish_processed_path()
        
        return result

    def imu_callback(self, msg):
        # Extract yaw from quaternion
        q = msg.orientation
        # Convert quaternion to Euler angles
        sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        sinp = 2 * (q.w * q.y - q.z * q.x)
        pitch = np.arcsin(sinp)
        
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.vehicle_yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        self.get_logger().debug(f'Vehicle yaw: {np.degrees(self.vehicle_yaw):.2f} degrees')

    def gnss_callback(self, msg):
        # Convert lat/lon to local coordinates (simplified)
        # In a real implementation, you would use a proper conversion
        self.vehicle_x = msg.longitude
        self.vehicle_y = msg.latitude
        self.get_logger().debug(f'Vehicle position: ({self.vehicle_x:.6f}, {self.vehicle_y:.6f})')
    
    def wheelspeed_callback(self, msg):
        self.vehicle_speed = msg.data  # in m/s
        self.get_logger().debug(f'Vehicle speed: {self.vehicle_speed:.2f} m/s')
    
    def steering_angle_callback(self, msg):
        # Current steering angle (for monitoring)
        self.last_steering_angle = msg.data
    
    def lane_coordinates_callback(self, msg):
        # Parse JSON string from the lane detection node
        try:
            # The message is a JSON string containing lane points
            lane_data = json.loads(msg.data)
            self.raw_waypoints = []
            
            if len(lane_data) > 0:
                # Select the center lane for path following
                # In this case we'll use the middle lane if multiple lanes are detected
                lane_idx = min(len(lane_data) // 2, len(lane_data) - 1)
                lane = lane_data[lane_idx]
                
                # Convert lane points to waypoints
                for point in lane:
                    if len(point) == 2:  # Ensure valid point [x, y]
                        self.raw_waypoints.append((point[0], point[1]))
                
                if self.raw_waypoints:
                    # Apply path smoothing
                    self.waypoints = self.smooth_path(self.raw_waypoints, self.smoothing_window)
                    self.reference_path_received = True
                    self.get_logger().info(f'Received lane data with {len(self.waypoints)} processed waypoints')
                    
                    # Publish the processed path for visualization
                    self.publish_processed_path()
            else:
                self.get_logger().warn('Received empty lane data')
                
        except json.JSONDecodeError:
            self.get_logger().error('Failed to parse lane coordinates JSON')
        except Exception as e:
            self.get_logger().error(f'Error processing lane data: {str(e)}')
    
    def smooth_path(self, waypoints, window_size):
        """Apply a simple moving average to smooth the path"""
        if len(waypoints) < window_size:
            return waypoints
            
        smoothed_path = []
        for i in range(len(waypoints)):
            # Calculate window bounds
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(waypoints), i + window_size // 2 + 1)
            
            # Calculate average x and y
            x_avg = sum(wp[0] for wp in waypoints[start_idx:end_idx]) / (end_idx - start_idx)
            y_avg = sum(wp[1] for wp in waypoints[start_idx:end_idx]) / (end_idx - start_idx)
            
            smoothed_path.append((x_avg, y_avg))
            
        return smoothed_path
    
    def publish_processed_path(self):
        """Publish the processed path for visualization"""
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        for x, y in self.waypoints:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
            
        self.path_publisher.publish(path_msg)
    
    def find_closest_waypoint(self):
        if not self.waypoints:
            return None
        
        # Find the closest waypoint to current vehicle position
        distances = [(self.vehicle_x - wp[0])**2 + (self.vehicle_y - wp[1])**2 for wp in self.waypoints]
        closest_idx = np.argmin(distances)
        
        return closest_idx
    
    def find_lookahead_point(self):
        """Find a point on the path ahead of the vehicle"""
        if not self.waypoints or len(self.waypoints) < 2:
            return None
            
        closest_idx = self.find_closest_waypoint()
        if closest_idx is None:
            return None
            
        # Start from closest point and find a point that's approximately
        # the lookahead distance away
        cumulative_dist = 0.0
        for i in range(closest_idx, len(self.waypoints) - 1):
            wp1 = self.waypoints[i]
            wp2 = self.waypoints[i + 1]
            segment_dist = np.sqrt((wp2[0] - wp1[0])**2 + (wp2[1] - wp1[1])**2)
            
            if cumulative_dist + segment_dist >= self.lookahead_distance:
                # Interpolate between wp1 and wp2 to get exact lookahead point
                overshoot = self.lookahead_distance - cumulative_dist
                ratio = overshoot / segment_dist if segment_dist > 0 else 0
                
                x = wp1[0] + ratio * (wp2[0] - wp1[0])
                y = wp1[1] + ratio * (wp2[1] - wp1[1])
                return (x, y)
                
            cumulative_dist += segment_dist
            
        # If we can't find a point far enough, return the last waypoint
        return self.waypoints[-1]

    def compute_steering_angle(self):
        if not self.reference_path_received or not self.waypoints:
            self.get_logger().warning('No reference path available')
            return 0.0
        
        # Using lookahead point for better performance
        lookahead_point = self.find_lookahead_point()
        if lookahead_point is None:
            # Fallback to closest point method
            closest_idx = self.find_closest_waypoint()
            if closest_idx is None:
                return 0.0
                
            # Get the closest waypoint
            closest_waypoint = self.waypoints[closest_idx]
            
            # Get the next waypoint for heading
            next_idx = min(closest_idx + 1, len(self.waypoints) - 1)
            if next_idx == closest_idx:  # We're at the last waypoint
                if closest_idx > 0:
                    next_idx = closest_idx
                    closest_idx = closest_idx - 1
                    closest_waypoint = self.waypoints[closest_idx]
                else:
                    return 0.0  # Can't determine path direction
            
            next_waypoint = self.waypoints[next_idx]
            
            # Calculate path heading
            path_dx = next_waypoint[0] - closest_waypoint[0]
            path_dy = next_waypoint[1] - closest_waypoint[1]
            path_yaw = np.arctan2(path_dy, path_dx)
            
            # Transform to vehicle coordinates
            dx = closest_waypoint[0] - self.vehicle_x
            dy = closest_waypoint[1] - self.vehicle_y
        else:
            # Use lookahead point
            # Find the closest waypoint for path direction
            closest_idx = self.find_closest_waypoint()
            if closest_idx is None or closest_idx >= len(self.waypoints) - 1:
                return 0.0
                
            # Get direction from closest to next waypoint
            wp1 = self.waypoints[closest_idx]
            wp2 = self.waypoints[min(closest_idx + 1, len(self.waypoints) - 1)]
            
            path_dx = wp2[0] - wp1[0]
            path_dy = wp2[1] - wp1[1]
            path_yaw = np.arctan2(path_dy, path_dx)
            
            # Use lookahead point for CTE calculation
            dx = lookahead_point[0] - self.vehicle_x
            dy = lookahead_point[1] - self.vehicle_y
            
        # Calculate cross track error (CTE)
        # This is the lateral distance from the path
        cte = dy * np.cos(self.vehicle_yaw) - dx * np.sin(self.vehicle_yaw)
        
        # Calculate heading error
        heading_error = np.arctan2(np.sin(path_yaw - self.vehicle_yaw), 
                                  np.cos(path_yaw - self.vehicle_yaw))
        
        # Stanley controller formula
        # Ensure minimum speed to avoid division by zero
        speed = max(self.vehicle_speed, 1.0)  # Minimum 1 m/s
        stanley_term = np.arctan(self.k * cte / speed)
        
        # Current time for PID
        current_time = time()
        dt = current_time - self.prev_time
        self.prev_time = current_time
        
        # PID correction based on CTE
        pid_correction = np.clip(self.pid_controller.compute(cte, dt), 
                                -np.radians(10), np.radians(10))
        
        # Combine Stanley and PID
        steering_angle = heading_error + stanley_term + pid_correction
        steering_angle = np.clip(steering_angle, -np.radians(30), np.radians(30))
        
        # Log data for plotting
        self.time_list.append(current_time - self.start_time)
        self.steering_angle_list.append(np.degrees(steering_angle))
        self.cte_list.append(cte)
        
        self.get_logger().debug(
            f"CTE: {cte:.2f}, Heading Error: {np.degrees(heading_error):.2f}°, "
            f"PID Correction: {np.degrees(pid_correction):.2f}°, "
            f"Final Steering: {np.degrees(steering_angle):.2f}°"
        )
        
        return steering_angle

    def control_loop(self):
        # Calculate steering angle using Stanley controller
        steering_angle = self.compute_steering_angle()
        
        # Create control message
        control_msg = Twist()
        control_msg.angular.z = steering_angle  # Steering in rad
        control_msg.linear.x = 0.3  # Throttle (fixed for now)
        
        # Publish control command
        self.control_publisher.publish(control_msg)
    
    def save_plot_data(self):
        # Save data for offline plotting
        data = {
            'time': self.time_list,
            'steering_angle': self.steering_angle_list,
            'cte': self.cte_list
        }
        
        try:
            with open('stanley_controller_data.json', 'w') as f:
                json.dump(data, f)
            self.get_logger().info('Saved controller data to stanley_controller_data.json')
        except Exception as e:
            self.get_logger().error(f'Failed to save data: {e}')
            
    def plot_data(self):
        # Generate plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot steering angle
        ax1.plot(self.time_list, self.steering_angle_list, 'b-', label='Steering Angle')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Steering Angle (°)')
        ax1.set_title('Steering Angle vs Time')
        ax1.grid(True)
        ax1.legend()
        
        # Plot CTE
        ax2.plot(self.time_list, self.cte_list, 'r-', label='Cross Track Error')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Error (m)')
        ax2.set_title('Cross Track Error vs Time')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('stanley_controller_performance.png')
        self.get_logger().info('Saved plots to stanley_controller_performance.png')

def main(args=None):
    rclpy.init(args=args)
    node = StanleyControllerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, saving data...')
        node.save_plot_data()
        node.plot_data()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()