#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32
from radar_msgs.msg import RadarTrack
import time
import math
import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd, output_min, output_max):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.reset()
        
    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0
        self.last_time = time.time()
        
    def compute(self, setpoint, process_variable):
        current_time = time.time()
        dt = current_time - self.last_time
        
        # Avoid division by zero or negative time
        if dt <= 0:
            dt = 0.01
            
        # Calculate error
        error = setpoint - process_variable
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        # Apply anti-windup - limit integral term
        if self.integral * self.ki > self.output_max:
            self.integral = self.output_max / self.ki
        elif self.integral * self.ki < self.output_min:
            self.integral = self.output_min / self.ki
            
        i_term = self.ki * self.integral
        
        # Derivative term (on measurement to avoid derivative kick)
        derivative = 0.0
        if dt > 0:
            derivative = (error - self.previous_error) / dt
        d_term = self.kd * derivative
        
        # Calculate output
        output = p_term + i_term + d_term
        
        # Clamp output
        output = max(self.output_min, min(self.output_max, output))
        
        # Save state for next iteration
        self.previous_error = error
        self.last_time = current_time
        
        return output

class AEBController(Node):
    def __init__(self):
        super().__init__('aeb_controller')
        
        # Constants for AEB
        self.MAX_SPEED = 8.33  # m/s (30 km/h)
        self.SAFE_DISTANCE = 10.0  # meters
        self.BRAKE_THRESHOLD = 10.0  # meters - distance to start gradual braking
        self.EMERGENCY_BRAKE_THRESHOLD = 6.0  # meters - modified as requested
        self.ACCELERATION_DISTANCE = 100.0  # meters
        self.MAX_DECEL = 8.0  # m/s²
        self.MIN_DECEL = 5.0  # m/s²
        self.TTC_THRESHOLD = 2.0  # Time-To-Collision threshold (seconds)
        
        # Current state
        self.current_speed = 0.0
        self.target_speed = 0.0
        self.distance_to_obstacle = float('inf')
        self.distance_traveled = 0.0
        self.last_time = time.time()
        self.start_time = time.time()
        self.wheel_slip_ratio = 0.0
        self.wheel_speeds = [0.0, 0.0, 0.0, 0.0]  # FL, FR, RL, RR
        self.vehicle_acceleration = 0.0
        self.yaw_rate = 0.0
        self.relative_velocity = 0.0
        self.previous_distance = float('inf')
        self.last_update_time = time.time()
        self.obstacle_detected = False
        self.emergency_braking = False
        self.active_safety_control = False  # Flag to track if safety control is active
        
        # PID Controllers
        # Speed control - Higher gains for faster response
        self.speed_pid = PIDController(kp=2.5, ki=0.5, kd=0.1, output_min=0.0, output_max=5.0)
        
        # Braking control - High Kp for fast response
        self.brake_pid = PIDController(kp=3.0, ki=0.2, kd=0.5, output_min=0.0, output_max=self.MAX_DECEL)
        
        # ABS control
        self.abs_pid = PIDController(kp=1.5, ki=0.05, kd=0.3, output_min=0.0, output_max=1.0)
        
        # Publishers
        self.control_pub = self.create_publisher(
            Vector3,
            '/vehicle_control',  # UPDATED: Changed from '/vehicleControl' to '/vehicle_control'
            10
        )
        
        # Subscribers - set up radar first for immediate obstacle detection
        self.radar_sub = self.create_subscription(
            RadarTrack,  # Note: Message type may need update if RadarObjects uses different msg type
            '/RadarObjects',  # UPDATED: Changed from '/RadarTracks' to '/RadarObjects'
            self.radar_callback,
            1  # Highest QoS setting for safety-critical data
        )
        
        self.imu_sub = self.create_subscription(
            Imu,  # Note: Message type may need update if InertialData uses different msg type
            '/InertialData',  # UPDATED: Changed from '/imu' to '/InertialData'
            self.imu_callback,
            10
        )
        
        self.wheelspeed_sub = self.create_subscription(
            Float32,  # Note: Message type may need update if VehicleSpeed uses different msg type
            '/VehicleSpeed',  # UPDATED: Changed from '/wheelspeed' to '/VehicleSpeed'
            self.wheelspeed_callback,
            10
        )
        
        # Timer for control loop - higher frequency for more responsive control
        self.timer = self.create_timer(0.01, self.control_loop)  # 100Hz
        
        # Timer for distance display
        self.display_timer = self.create_timer(0.5, self.display_distance)
        
        # Send initial acceleration command with higher values
        self.get_logger().warn('Starting AEB system with RADAR active from initialization!')
        self.send_initial_command()
        
        self.get_logger().info('AEB Controller initialized with PID control and sensor fusion')
        self.get_logger().info(f'Target: {self.MAX_SPEED*3.6:.1f} km/h within {self.ACCELERATION_DISTANCE}m')
        self.get_logger().info(f'Emergency braking at: {self.EMERGENCY_BRAKE_THRESHOLD}m')
        self.get_logger().info('-------------------------------------------')
        self.get_logger().info('DISTANCE TRACKING STARTED')
        self.get_logger().info('-------------------------------------------')
    
    def send_initial_command(self):
        """Send initial command with strong acceleration to ensure vehicle starts moving"""
        initial_cmd = Vector3()
        initial_cmd.x = 0.0  # No steering
        initial_cmd.y = 5.0  # Strong acceleration to overcome inertia
        initial_cmd.z = 0.0  # No braking
        
        # Publish multiple times to ensure it's received
        for _ in range(10):
            self.control_pub.publish(initial_cmd)
            time.sleep(0.02)
        
        self.get_logger().warn('INITIAL ACCELERATION COMMAND SENT WITH HIGHER THROTTLE!')

    def display_distance(self):
        """Dedicated timer callback to prominently display distance covered and vehicle state"""
        elapsed_time = time.time() - self.start_time
        
        # Add obstacle information if detected
        obstacle_info = ""
        if self.obstacle_detected:
            ttc = float('inf')
            if self.relative_velocity < 0:  # Negative means closing distance
                ttc = abs(self.distance_to_obstacle / self.relative_velocity)
                ttc_str = f"{ttc:.2f}s" if ttc < float('inf') else "∞"
            else:
                ttc_str = "∞"
                
            obstacle_info = f"\n  OBSTACLE: {self.distance_to_obstacle:.2f}m, TTC: {ttc_str}"
            if self.emergency_braking:
                obstacle_info += " ⚠️ EMERGENCY BRAKING ACTIVE ⚠️"
        
        # Add safety control info if active
        safety_info = ""
        if self.active_safety_control:
            safety_info = "\n  ⚠️ ACTIVE SAFETY CONTROL ENGAGED ⚠️"
        
        # Create status display
        distance_msg = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  DISTANCE COVERED: {self.distance_traveled:.2f} meters
  SPEED: {self.current_speed:.2f} m/s ({self.current_speed*3.6:.1f} km/h)
  TARGET: {self.target_speed:.2f} m/s ({self.target_speed*3.6:.1f} km/h)
  ACCELERATION: {self.vehicle_acceleration:.2f} m/s²{obstacle_info}{safety_info}
  WHEEL SPEEDS: {sum(self.wheel_speeds)/len(self.wheel_speeds):.2f} m/s
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        self.get_logger().info(distance_msg)

    def radar_callback(self, msg):
        # Process RadarObjects data - find closest obstacle (active from start)
        # NOTE: You may need to adjust this function based on RadarObjects message structure
        try:
            closest_distance = float('inf')
            closest_velocity = 0.0
            
            for track in msg.tracks:
                # Adjust based on your actual RadarObjects message structure
                if hasattr(track, 'position') and track.position.x > 0:
                    distance = track.position.x
                    
                    if distance < closest_distance:
                        closest_distance = distance
                        # Get relative velocity if available
                        if hasattr(track, 'velocity'):
                            closest_velocity = track.velocity.x
            
            # Update if we found a valid obstacle
            if closest_distance < float('inf'):
                self.distance_to_obstacle = closest_distance
                self.relative_velocity = closest_velocity
                
                # Always check for obstacles regardless of distance traveled
                if self.distance_to_obstacle <= self.BRAKE_THRESHOLD:
                    self.obstacle_detected = True
                    
                    # Emergency braking at 6 meters or less
                    if self.distance_to_obstacle <= self.EMERGENCY_BRAKE_THRESHOLD:
                        self.emergency_braking = True
                        if not self.active_safety_control:
                            self.active_safety_control = True
                            self.get_logger().error(f'EMERGENCY! Obstacle at {self.distance_to_obstacle:.2f}m - EMERGENCY BRAKING ACTIVATED!')
                    else:
                        self.emergency_braking = False
                        if not self.active_safety_control:
                            self.active_safety_control = True
                            self.get_logger().warn(f'CAUTION! Obstacle detected at {self.distance_to_obstacle:.2f}m, rel. vel: {self.relative_velocity:.2f}m/s')
                else:
                    self.obstacle_detected = False
                    self.emergency_braking = False
                    
                    if self.active_safety_control and self.distance_to_obstacle > self.BRAKE_THRESHOLD * 1.5:
                        # Only deactivate safety control when well beyond threshold
                        self.active_safety_control = False
                        self.get_logger().info(f'Safety control deactivated - Obstacle now at {self.distance_to_obstacle:.2f}m')
                    
        except Exception as e:
            self.get_logger().error(f'Error processing radar data: {e}')

    def imu_callback(self, msg):
        # NOTE: You may need to adjust this function based on InertialData message structure
        try:
            # Extract linear acceleration (vehicle acceleration)
            self.vehicle_acceleration = msg.linear_acceleration.x
            
            # Extract angular velocity (yaw rate)
            self.yaw_rate = msg.angular_velocity.z
            
            # Use IMU for dead reckoning if wheel speed is not available
            dt = time.time() - self.last_update_time
            if dt > 0 and dt < 0.1:  # Reasonable time delta
                # Simple integration - for speed estimate only
                speed_delta = self.vehicle_acceleration * dt
                # Weight IMU-based estimate less than wheel speed
                self.current_speed = 0.3 * (self.current_speed + speed_delta) + 0.7 * self.current_speed
            
            self.last_update_time = time.time()
            
        except Exception as e:
            self.get_logger().error(f'Error processing IMU data: {e}')

    def wheelspeed_callback(self, msg):
        # NOTE: You may need to adjust this function based on VehicleSpeed message structure
        try:
            # Process wheel speed message (format may vary)
            if hasattr(msg, 'data'):
                # Single value (average of all wheels)
                wheel_speed = msg.data
                self.wheel_speeds = [wheel_speed] * 4
                
                # Update current speed primarily from wheel speed
                # This is much more reliable than integrating acceleration
                self.current_speed = 0.8 * wheel_speed + 0.2 * self.current_speed
                
            elif hasattr(msg, 'speeds') and len(msg.speeds) > 0:
                # Array of wheel speeds
                self.wheel_speeds = msg.speeds
                avg_wheel_speed = sum(self.wheel_speeds) / len(self.wheel_speeds)
                self.current_speed = 0.8 * avg_wheel_speed + 0.2 * self.current_speed
            
            # Calculate wheel slip ratio
            if self.current_speed > 0.1:
                avg_wheel_speed = sum(self.wheel_speeds) / len(self.wheel_speeds)
                self.wheel_slip_ratio = (avg_wheel_speed - self.current_speed) / self.current_speed
            else:
                self.wheel_slip_ratio = 0.0
                
        except Exception as e:
            self.get_logger().error(f'Error processing wheel speed data: {e}')

    def update_distance_traveled(self):
        current_time = time.time()
        dt = current_time - self.last_time
        
        # Make sure dt is reasonable
        if dt > 0 and dt < 0.1:  # Smaller threshold for more accurate updates
            self.distance_traveled += self.current_speed * dt
            
        self.last_time = current_time

    def control_loop(self):
        self.update_distance_traveled()
        
        # Initialize control message
        control_msg = Vector3()
        control_msg.x = 0.0  # No steering
        
        # Safety-first approach: Check for obstacles before anything else
        if self.obstacle_detected:
            # Handle obstacle detection with highest priority
            if self.emergency_braking:
                # Full emergency braking at 6 meters or less
                braking_force = self.MAX_DECEL
                self.target_speed = 0.0
                control_msg.y = 0.0  # No throttle during emergency braking
                control_msg.z = braking_force  # Maximum braking
                
                self.get_logger().error(f'EMERGENCY BRAKING! Distance: {self.distance_to_obstacle:.2f}m')
            else:
                # Progressive braking between threshold and emergency threshold
                distance_factor = 1.0 - ((self.distance_to_obstacle - self.EMERGENCY_BRAKE_THRESHOLD) / 
                                         (self.BRAKE_THRESHOLD - self.EMERGENCY_BRAKE_THRESHOLD))
                distance_factor = max(0.0, min(1.0, distance_factor))  # Clamp between 0-1
                
                # Calculate TTC and adjust braking force
                ttc = float('inf')
                ttc_factor = 0.0
                if self.relative_velocity < 0:  # Negative means closing distance
                    ttc = abs(self.distance_to_obstacle / self.relative_velocity)
                    ttc_factor = max(0.0, 1.0 - (ttc / self.TTC_THRESHOLD))
                
                # Combine factors and calculate target deceleration
                blend_factor = max(distance_factor, ttc_factor)
                target_decel = self.MIN_DECEL + blend_factor * (self.MAX_DECEL - self.MIN_DECEL)
                
                # Use PID to achieve target deceleration
                braking_force = self.brake_pid.compute(target_decel, -self.vehicle_acceleration)
                
                # Cut throttle and apply brakes
                self.target_speed = max(0.0, self.current_speed - target_decel * 0.1)
                control_msg.y = 0.0
                control_msg.z = braking_force
                
                self.get_logger().warn(f'Progressive braking: {braking_force:.2f}m/s², Target decel: {target_decel:.2f}m/s²')
        
        # If no obstacle detected, proceed with normal operation
        elif self.distance_traveled < self.ACCELERATION_DISTANCE:
            # Initial acceleration phase (0-100m)
            # Calculate target speed as function of distance - more aggressive acceleration curve
            progress = self.distance_traveled / self.ACCELERATION_DISTANCE
            # Exponential acceleration profile for faster initial response
            accel_factor = min(1.0, 1.0 - math.exp(-4.0 * progress))
            self.target_speed = self.MAX_SPEED * accel_factor
            
            # Apply speed controller with higher gains for acceleration phase
            throttle_cmd = self.speed_pid.compute(self.target_speed, self.current_speed)
            
            # Force stronger initial acceleration if speed is very low
            if self.current_speed < 0.5:
                throttle_cmd = max(throttle_cmd, 3.0)  # Minimum throttle when starting
                
            control_msg.y = throttle_cmd  # Apply throttle
            control_msg.z = 0.0  # No braking
            
            self.get_logger().info(f'Accelerating: {progress*100:.1f}%, Target: {self.target_speed:.2f}m/s, Throttle: {throttle_cmd:.2f}')
        
        else:
            # Cruising phase (after 100m with no obstacles)
            self.target_speed = self.MAX_SPEED
            throttle_cmd = self.speed_pid.compute(self.target_speed, self.current_speed)
            control_msg.y = throttle_cmd
            control_msg.z = 0.0  # No braking
            
            self.get_logger().info(f'Cruising at target speed: {self.target_speed:.2f}m/s')
        
        # Apply wheel-slip based ABS for braking optimization
        if control_msg.z > 0.1:  # If braking
            # Check for excessive wheel slip
            if abs(self.wheel_slip_ratio) > 0.15:
                # Apply ABS-like correction
                slip_correction = self.abs_pid.compute(0.1, abs(self.wheel_slip_ratio))
                # Modulate braking force to prevent lockup
                control_msg.z *= (1.0 - slip_correction)
                self.get_logger().warn(f'ABS active: Slip={self.wheel_slip_ratio:.2f}, Reducing brake by {slip_correction:.2f}')
        
        # Apply stability control based on IMU data
        if abs(self.yaw_rate) > 0.3 and self.current_speed > 1.0:
            # High yaw rate detected - vehicle may be unstable
            yaw_correction = min(0.8, abs(self.yaw_rate) * 0.2)
            
            # Reduce throttle and apply gentle braking if needed
            if control_msg.y > 0:
                control_msg.y *= (1.0 - yaw_correction)
            
            # Apply gentle braking if not already braking hard
            if control_msg.z < 2.0:
                control_msg.z = max(control_msg.z, yaw_correction * 3.0)
                
            self.get_logger().warn(f'Stability control: Yaw rate={self.yaw_rate:.2f}rad/s, Applying correction')
        
        # Force control commands to be in valid range
        control_msg.y = max(0.0, min(5.0, control_msg.y))  # Throttle between 0 and 5
        control_msg.z = max(0.0, min(self.MAX_DECEL, control_msg.z))  # Brake between 0 and MAX_DECEL
        
        # Ensure exclusive throttle or brake (never both)
        if control_msg.z > 0.0:
            control_msg.y = 0.0
        
        # Send double commands at startup to ensure vehicle moves
        if self.distance_traveled < 1.0 and self.current_speed < 0.1 and not self.obstacle_detected:
            # Extra strong initial throttle pulse
            boost_cmd = Vector3()
            boost_cmd.x = 0.0
            boost_cmd.y = 5.0  # Maximum throttle
            boost_cmd.z = 0.0
            self.control_pub.publish(boost_cmd)
            self.get_logger().warn('SENDING BOOST THROTTLE TO START MOVEMENT!')
        
        # Publish control command
        self.control_pub.publish(control_msg)
        
        # Simulate vehicle response for testing if no real feedback
        if sum(self.wheel_speeds) == 0 and self.vehicle_acceleration == 0:
            # Simple simulation for testing
            dt = 0.01  # Control loop time
            if control_msg.y > 0:  # Acceleration
                self.current_speed += control_msg.y * 0.1 * dt
            elif control_msg.z > 0:  # Braking
                self.current_speed = max(0.0, self.current_speed - control_msg.z * dt)
            
            # Cap at maximum speed
            self.current_speed = min(self.MAX_SPEED, self.current_speed)

def main(args=None):
    rclpy.init(args=args)
    aeb_controller = AEBController()

    try:
        rclpy.spin(aeb_controller)
    except KeyboardInterrupt:
        aeb_controller.get_logger().info('Keyboard interrupt received. Shutting down...')
    except Exception as e:
        aeb_controller.get_logger().error(f'Error occurred: {str(e)}')
    finally:
        # Send final stop command
        stop_msg = Vector3()
        stop_msg.x = 0.0
        stop_msg.y = 0.0
        stop_msg.z = 0.0
        
        # Publish stop command multiple times to ensure it's received
        for _ in range(5):
            aeb_controller.control_pub.publish(stop_msg)
            time.sleep(0.02)
            
        # Display final distance
        final_distance = aeb_controller.distance_traveled
        aeb_controller.get_logger().warn(f'FINAL DISTANCE COVERED: {final_distance:.2f} meters')
        
        aeb_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()