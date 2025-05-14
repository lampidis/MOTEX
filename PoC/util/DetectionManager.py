import cv2
import numpy as np
import torch
from filterpy.kalman import KalmanFilter


class DetectionsManager:
    def __init__(self, fps=30, real_car_width=1.8, filter_size=5):
            self.kalman_filters = {}
            self.fps = fps
            self.real_car_width = real_car_width  
            self.velocity_history = {}  
            self.filter_size = filter_size  
            self.last_positions = {}


    def smooth_velocity(self, tracker_id, vx, vy, vz):
        """Apply a moving average filter over last `N` frames to smooth velocity estimates."""
        if tracker_id not in self.velocity_history:
            self.velocity_history[tracker_id] = []  # Initialize list
        
        # ✅ Append new velocity measurement
        self.velocity_history[tracker_id].append((vx, vy, vz))

        # ✅ Keep only last `N` frames
        if len(self.velocity_history[tracker_id]) > self.filter_size:
            self.velocity_history[tracker_id].pop(0)  # Remove oldest entry

        # ✅ Compute average velocity
        avg_vx = np.mean([v[0] for v in self.velocity_history[tracker_id]])
        avg_vy = np.mean([v[1] for v in self.velocity_history[tracker_id]])
        avg_vz = np.mean([v[2] for v in self.velocity_history[tracker_id]])

        return avg_vx, avg_vy, avg_vz

    def estimate_depth(self, bbox_width_pixels: float) -> float:
        """Estimate depth (Z) based on bounding box width and focal length."""
        if bbox_width_pixels == 0:
            return 0  # Prevent division by zero
        # return (fx * self.real_car_width) / bbox_width_pixels
        return (self.real_car_width) / bbox_width_pixels

    def initialize_kalman_filter(self, tracker_id, x, y, z):
            """Initializes a Kalman filter for tracking position and velocity."""
            print(f"🔹 Initializing Kalman Filter for Tracker {tracker_id}")

            kf = KalmanFilter(dim_x=9, dim_z=3)  # 9D state (x, y, z, vx, vy, vz, ax, ay, az)
            dt = 1 / self.fps  

            # ✅ State transition matrix (Constant Acceleration Model)
            kf.F = np.array([
                [1, 0, 0, dt, 0, 0, 0.5 * dt ** 2, 0, 0], 
                [0, 1, 0, 0, dt, 0, 0, 0.5 * dt ** 2, 0],  
                [0, 0, 1, 0, 0, dt, 0, 0, 0.5 * dt ** 2],  
                [0, 0, 0, 1, 0, 0, dt, 0, 0],  
                [0, 0, 0, 0, 1, 0, 0, dt, 0],  
                [0, 0, 0, 0, 0, 1, 0, 0, dt],  
                [0, 0, 0, 0, 0, 0, 1, 0, 0],  
                [0, 0, 0, 0, 0, 0, 0, 1, 0],  
                [0, 0, 0, 0, 0, 0, 0, 0, 1]   
            ])

            # ✅ Measurement matrix (Observing x, y, z only)
            kf.H = np.array([
                [1, 0, 0, 0, 0, 0, 0, 0, 0],  
                [0, 1, 0, 0, 0, 0, 0, 0, 0],  
                [0, 0, 1, 0, 0, 0, 0, 0, 0]   
            ])

            kf.Q = np.eye(9) * 1e-3  
            kf.R = np.eye(3) * 5  
            kf.P *= 500  
            kf.x = np.array([[x], [y], [z], [0], [0], [0], [0], [0], [0]])  

            self.kalman_filters[tracker_id] = kf
            self.last_positions[tracker_id] = (x, y, z)

    def update(self, detections):
            velocities = {}

            for tracker_id, (x1, y1, x2, y2) in zip(detections.tracker_id, detections.xyxy):
                x, y = (x1 + x2) // 2, (y1 + y2) // 2
                bbox_width = abs(x2 - x1)

                # ✅ Estimate depth (Z)
                z = (1.8 * 600) / bbox_width if bbox_width > 0 else 0

                if tracker_id not in self.kalman_filters:
                    self.initialize_kalman_filter(tracker_id, x, y, z)
                else:
                    kf = self.kalman_filters[tracker_id]
                    kf.predict()  # ✅ Kalman Filter Prediction Step
                    measurement = np.array([[x], [y], [z]])
                    kf.update(measurement)

                    prev_x, prev_y, prev_z = self.last_positions.get(tracker_id, (x, y, z))
                    vx = (x - prev_x) * self.fps
                    vy = (y - prev_y) * self.fps
                    vz = (z - prev_z) * self.fps

                    velocities[tracker_id] = (vx, vy, vz)
                    self.last_positions[tracker_id] = (x, y, z)

            return detections, velocities

    
    def predict_future_position(self, tracker_id: int, prediction_time: float = 2.0):
        """Predicts future position using velocity and acceleration from Kalman Filter."""
        
        if tracker_id not in self.kalman_filters or tracker_id not in self.last_positions:
            return None  # No data available for prediction

        # ✅ Get last known position
        x, y, z = self.last_positions[tracker_id]  

        # ✅ Get **smoothed** velocity
        vx, vy, vz = self.smooth_velocity(tracker_id, 
                                        self.kalman_filters[tracker_id].x[3, 0],  
                                        self.kalman_filters[tracker_id].x[4, 0],  
                                        self.kalman_filters[tracker_id].x[5, 0])

        # ✅ Extract acceleration from Kalman Filter
        ax = self.kalman_filters[tracker_id].x[6, 0]  # X acceleration
        ay = self.kalman_filters[tracker_id].x[7, 0]  # Y acceleration
        az = self.kalman_filters[tracker_id].x[8, 0]  # Z acceleration

        # ✅ Predict future position using **kinematic equation** (s = s0 + v*t + 0.5*a*t²)
        future_x = x + vx * prediction_time + 0.5 * ax * (prediction_time ** 2)
        future_y = y + vy * prediction_time + 0.5 * ay * (prediction_time ** 2)
        future_z = z + vz * prediction_time + 0.5 * az * (prediction_time ** 2)

        # ✅ Ensure future_z is reasonable (prevents division errors)
        future_z = max(future_z, 0.1)

        return int(round(future_x)), int(round(future_y)), future_z  # Return (fx, fy, fz)

    def predict_future_bbox(self, tracker_id: int, current_bbox: tuple, prediction_time: float = 2.0):
        """Predicts the future bounding box centered at the predicted future position, 
        with scaling based on depth velocity (Vz)."""

        if tracker_id not in self.kalman_filters or tracker_id not in self.last_positions:
            return None  # No data available for prediction

        # ✅ Get current bounding box size
        x1, y1, x2, y2 = current_bbox
        current_width = x2 - x1
        current_height = y2 - y1

        # ✅ Predict future position
        future_pos = self.predict_future_position(tracker_id, prediction_time)
        if not future_pos:
            return None  # If no future position, return nothing

        fx, fy, fz = future_pos  # Extract predicted position

        # ✅ Extract current and future depth (Z-position)
        current_z = self.last_positions[tracker_id][2]
        future_z = fz  # Already computed in `predict_future_position`

        # ✅ Avoid division errors (ensure future_z is non-zero)
        future_z = max(future_z, 0.1)

        # ✅ Scale bounding box size based on depth change
        scale_factor = current_z / future_z  
        predicted_width = max(int(current_width * scale_factor), 10)  
        predicted_height = max(int(current_height * scale_factor), 10)

        # ✅ Center future bounding box at predicted position (`fx, fy`)
        future_x1 = int(fx - predicted_width // 2)
        future_x2 = int(fx + predicted_width // 2)
        future_y1 = int(fy - predicted_height // 2)
        future_y2 = int(fy + predicted_height // 2)

        return future_x1, future_y1, future_x2, future_y2
    
    def check_collision(predicted_bboxes):
        """
        Check if any two predicted bounding boxes intersect.
        Returns a set of tracker_ids involved in a collision and the collision area.
        """
        collision_ids = set()
        collision_areas = []  # Store (x_min, y_min, x_max, y_max) of collision areas

        for i, (id1, bbox1) in enumerate(predicted_bboxes.items()):
            x1_min, y1_min, x1_max, y1_max = bbox1

            for j, (id2, bbox2) in enumerate(predicted_bboxes.items()):
                if i >= j:
                    continue  # Avoid duplicate checks

                x2_min, y2_min, x2_max, y2_max = bbox2

                # ✅ Check if bounding boxes overlap
                if x1_min < x2_max and x1_max > x2_min and y1_min < y2_max and y1_max > y2_min:
                    collision_ids.add(id1)
                    collision_ids.add(id2)

                    # ✅ Compute intersection area (collision box)
                    collision_x_min = max(x1_min, x2_min)
                    collision_y_min = max(y1_min, y2_min)
                    collision_x_max = min(x1_max, x2_max)
                    collision_y_max = min(y1_max, y2_max)

                    collision_areas.append((collision_x_min, collision_y_min, collision_x_max, collision_y_max))

        return collision_ids, collision_areas  # Return involved tracker IDs & collision areas
