import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv
from util.LaneDetector import LaneDetector

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])

class VideoProcessor:
    def __init__(self, model_path="yolov8n.pt",
                 input_video_path="",
                 output_video_path="out.mp4",
                 confidence_threshold=0.3, 
                 iou_threshold=0.4,
                 save=False):
        
        self.model = YOLO(model_path).to(DEVICE)
        for param in self.model.parameters():
            param.grad = None
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.save = save
        self.tracker = sv.ByteTrack()
        
        # 📍 Visual tracking annotations
        self.trace_annotator = sv.TraceAnnotator(color=COLORS, position=sv.Position.CENTER, trace_length=50, thickness=2)
        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        self.label_annotator = sv.LabelAnnotator()

        # 🎥 Video paths
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path

    def process_video(self):
        cap = cv2.VideoCapture(self.input_video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video file {self.input_video_path}")
            return

        # 🎥 Get video properties
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 📝 Define video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        if self.save:
            out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (frame_width, frame_height))

        print(f"✅ Processing video: {self.input_video_path} → {self.output_video_path}")

        for i in tqdm(range(total_frames), desc="Processing Video"):
            # if i%2 == 0: continue
            ret, frame = cap.read()
            frame = cv2.resize(frame, (640,360))
            if not ret:
                break

            annotated_frame = self.process_frame(frame)
            if annotated_frame is not None:
                cv2.imshow("Real-Time Crash Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if self.save:
                out.write(annotated_frame)

        cap.release()
        if self.save:
            out.release()
        print(f"✅ Video processing completed. Output saved at {self.output_video_path}")

    def process_webcam(self):
        """Processes video feed from the webcam and overlays detected collisions."""
        cap = cv2.VideoCapture(0)  # Open webcam feed

        if not cap.isOpened():
            print(f"❌ ERROR: Could not access webcam.")
            return

        print(f"🎥 Webcam feed started!")

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to capture frame.")
                break

            frame_count += 1
            annotated_frame = self.process_frame(frame)

            if annotated_frame is not None:
                cv2.imshow("Real-Time Crash Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        print("🛑 Webcam stream ended.")
        cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
        """Process each frame: detect objects, filter cars, and detect collisions."""
        
        # detector = LaneDetector()
        # frame = detector.detect(frame)
        results = self.model(frame, verbose=False, conf=self.conf_threshold, iou=self.iou_threshold)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Ensure detections exist before filtering
        if len(detections.xyxy) == 0:
            return frame  # Skip processing if no objects detected
        
        h, w = frame.shape[:2]
        for box in results.boxes.data.tolist():
            x1, y1, x2, y2, score, cls = box
            box_height = y2 - y1
            box_width = x2 - x1
            rel_height = box_height / h
            rel_width = box_width / w
            
            # Too close → red, else green
            if rel_width > 0.1 or rel_height > 0.1:
                if x2/w > 0.3 or x2/w > 0.7:
                    color = (0, 0, 255) 
                else:
                    color = (0, 165, 255)
            else: color = (0, 255, 0)
            
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                thickness=2
            )
        
        # Apply tracking first to ensure `tracker_id` is available
        # detections = self.tracker.update_with_detections(detections)

        # # Ensure `tracker_id` is not None
        # if detections.tracker_id is None:
        #     return frame  # Skip processing if tracker IDs are missing

        # Filter for only "car" detections
        # car_detections = []
        # car_tracker_ids = []
        # for i, (det, label) in enumerate(zip(detections.xyxy, detections.class_id)):
        #     if self.model.names[label] == "car":  # Ensuring only cars are processed
        #         car_detections.append(det)
        #         car_tracker_ids.append(detections.tracker_id[i])  # ✅ Ensure tracker_id exists before appending

        # # If no cars are detected, return the frame normally
        # if len(car_detections) == 0:
        #     return frame

        # # Convert car detections back to NumPy array
        # detections.xyxy = np.array(detections)
        # detections.tracker_id = np.array(car_tracker_ids)

        # print(f"🔍 YOLO Detected {len(detections)} cars")

        return frame



    def annotate_frame(self, frame, detections):
        """Annotate frame with bounding boxes, traces, and highlight overlapping areas."""
        annotated_frame = frame.copy()
        if len(detections) == 0:
            return frame

        # Draw detection bounding boxes
        annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)

        # Store bounding boxes
        bounding_boxes = []
        for tracker_id, (x1, y1, x2, y2) in zip(detections.tracker_id, detections.xyxy):
            bounding_boxes.append((x1, y1, x2, y2))

        # Detect overlapping bounding boxes
        overlap_regions = []
        for i in range(len(bounding_boxes)):
            for j in range(i + 1, len(bounding_boxes)):
                x1, y1, x2, y2 = bounding_boxes[i]
                x1b, y1b, x2b, y2b = bounding_boxes[j]

                # Calculate intersection area
                xi1, yi1 = max(x1, x1b), max(y1, y1b)
                xi2, yi2 = min(x2, x2b), min(y2, y2b)

                if xi1 < xi2 and yi1 < yi2:  # Check for valid overlap
                    overlap_regions.append((xi1, yi1, xi2, yi2))

        # Draw a semi-transparent red overlay on overlapping areas
        overlay = annotated_frame.copy()
        alpha = 0.5  # Transparency level

        for (x1, y1, x2, y2) in overlap_regions:
            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), -1)

        # Blend overlay with the original frame
        cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)

        return annotated_frame