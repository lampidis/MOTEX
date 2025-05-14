import argparse
import cv2
import numpy as np
import time
from tqdm import tqdm
from util.VideoProcessor import VideoProcessor

# 🚗 Define YOLO vehicle class IDs (Car, Motorcycle, Bus, Truck)
VEHICLE_CLASS_IDS = {2, 3, 5, 7}



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Vehicle Tracking with YOLO and ByteTrack")
    parser.add_argument("--input_path", required=True, help="Path to input video file", type=str)
    parser.add_argument("--output_path", required=True, help="Path to save processed output video", type=str)

    args = parser.parse_args()

    processor = VideoProcessor(
        input_video_path=args.input_path,
    )

    processor.process_video()