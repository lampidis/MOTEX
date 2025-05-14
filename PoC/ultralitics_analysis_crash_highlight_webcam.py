import cv2
import numpy as np
import torch
from filterpy.kalman import KalmanFilter
import supervision as sv
from util.VideoProcessor import VideoProcessor

# from calibrate_camera import calibration_data  

# ✅ Extract calibration parameters
# camera_matrix = calibration_data["camera_matrix"]
# dist_coeffs = calibration_data["dist_coeffs"]
# fx, fy = calibration_data["fx"], calibration_data["fy"]
# cx, cy = calibration_data["cx"], calibration_data["cy"]



if __name__ == "__main__":
    video_processor = VideoProcessor("yolov8n.pt")  # Use standard YOLOv8 Nano model
    video_processor.process_webcam()