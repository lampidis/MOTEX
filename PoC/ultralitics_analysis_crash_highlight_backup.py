import cv2
import numpy as np
import torch
from filterpy.kalman import KalmanFilter
import supervision as sv

# from calibrate_camera import calibration_data
from util.VideoProcessor import VideoProcessor

# ✅ Extract calibration parameters
# camera_matrix = calibration_data["camera_matrix"]
# dist_coeffs = calibration_data["dist_coeffs"]
# fx, fy = calibration_data["fx"], calibration_data["fy"]
# cx, cy = calibration_data["cx"], calibration_data["cy"]



def annotate_frame(frame, detections, velocities):
    annotated_frame = frame.copy()

    if len(detections) > 0:
        detections.class_id = np.zeros(len(detections), dtype=int)  

        # ✅ Draw current bounding boxes
        annotated_frame = self.box_annotator.annotate(annotated_frame, detections)

        # ✅ Draw past traces
        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)

        # ✅ Store predicted bounding boxes for collision checking
        predicted_bboxes = {}

        frame_height, frame_width, _ = frame.shape  # Get frame size

        for tracker_id, (x1, y1, x2, y2) in zip(detections.tracker_id, detections.xyxy):
            # ✅ Calculate center of bounding box
            x, y = (x1 + x2) // 2, (y1 + y2) // 2  

            # ✅ Get velocity (smoothed)
            vx, vy, vz = velocities.get(tracker_id, (0, 0, 0))

            # ✅ Predict Future Position
            future_pos = self.detections_manager.predict_future_position(tracker_id, 2)  # Predict 2 sec ahead
            if future_pos:
                fx, fy, fz = future_pos
                fx, fy = int(fx), int(fy)  # Ensure integers

                # ✅ Ensure predicted position is within frame bounds
                fx = max(0, min(frame_width - 1, fx))
                fy = max(0, min(frame_height - 1, fy))

                # 🔵 Draw the predicted position as a blue dot
                cv2.circle(annotated_frame, (fx, fy), 6, (255, 0, 0), -1)
                cv2.putText(annotated_frame, "Predicted", (fx + 5, fy - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

                # ✅ Predict Future Bounding Box
                future_bbox = self.detections_manager.predict_future_bbox(tracker_id, (x1, y1, x2, y2), 2)
                if future_bbox:
                    fx1, fy1, fx2, fy2 = future_bbox
                    fx1, fy1, fx2, fy2 = int(fx1), int(fy1), int(fx2), int(fy2)  # Ensure integers

                    # ✅ Ensure future bounding box is within frame bounds
                    fx1 = max(0, min(frame_width - 1, fx1))
                    fy1 = max(0, min(frame_height - 1, fy1))
                    fx2 = max(0, min(frame_width - 1, fx2))
                    fy2 = max(0, min(frame_height - 1, fy2))

                    # 🔵 Draw the predicted bounding box in blue
                    cv2.rectangle(annotated_frame, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)
                    cv2.putText(annotated_frame, "Predicted Size", (fx1, fy1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

                    # ✅ Store predicted bbox for collision detection
                    predicted_bboxes[tracker_id] = (fx1, fy1, fx2, fy2)

                # 🔵 Draw a purple line from current position to predicted position
                cv2.line(annotated_frame, (x, y), (fx, fy), (255, 0, 255), 2)

            # ✅ Display Speed Information
            speed_x = round(abs(vx) * 0.036, 2)  # Convert to km/h
            speed_y = round(abs(vy) * 0.036, 2)
            speed_z = round(abs(vz) * 0.036, 2)  # Depth velocity

            # ✅ Convert Z-depth (distance from camera) to meters
            depth_m = round(fz, 2)

            # 🟡 Draw velocity vector (indicating movement direction)
            arrow_length = 20
            arrow_end_x = int(x + vx * arrow_length)
            arrow_end_y = int(y + vy * arrow_length)

            cv2.arrowedLine(annotated_frame, (x, y), (arrow_end_x, arrow_end_y), (0, 255, 255), 2, tipLength=0.3)

            # 🟡 Draw velocity and depth text
            cv2.putText(annotated_frame, f"ID {tracker_id} | Vx: {speed_x} km/h | Vy: {speed_y} km/h | Vz: {speed_z} km/h | Z: {depth_m}m",
                        (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # ✅ Check for collisions
        collision_ids, collision_areas = self.detections_manager.check_collision(predicted_bboxes)

        for tracker_id in detections.tracker_id:
            x1, y1, x2, y2 = detections.xyxy[detections.tracker_id == tracker_id][0]

            # 🚨 Highlight vehicles predicted to collide in RED
            color = (0, 255, 0)  # Default GREEN for normal tracking
            if tracker_id in collision_ids:
                color = (0, 0, 255)  # 🚨 RED if collision predicted

            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # 🔴 Draw collision areas as red warning boxes
        for (cx1, cy1, cx2, cy2) in collision_areas:
            cv2.rectangle(annotated_frame, (cx1, cy1), (cx2, cy2), (0, 0, 255), 3)  # 🔴 Collision area
            cv2.putText(annotated_frame, "Collision!", (cx1, cy1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    return annotated_frame

if __name__ == "__main__":
    model_path = r"D:\Programs\VisualStudio\MOTEX\CAR_WEBCAM_SEG\supervision\examples\traffic_analysis\data\traffic_analysis.pt"  # ✅ Replace with your YOLO model path
    video_path = r"D:\Programs\VisualStudio\MOTEX\CAR_WEBCAM_SEG\supervision\examples\traffic_analysis\testvideocarscrashing1.mp4"  # ✅ Replace with your video file path
    output_path = r"D:\Programs\VisualStudio\MOTEX\CAR_WEBCAM_SEG\supervision\examples\traffic_analysis\output_testvideocarscrashing1_highlight.mp4"  # ✅ (Optional) Save processed video

    video_processor = VideoProcessor(model_path, video_path, output_path)
    video_processor.process_video()