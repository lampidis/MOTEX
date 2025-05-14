# face_detection_yolov8_hf.py
"""
YOLOv8 Face Detection using Hugging Face pretrained weights

This script downloads a face-detection model fine-tuned on WIDER FACE from Hugging Face
(arnabdhar/YOLOv8-Face-Detection) and runs inference on images or video/webcam.

Requirements:
    pip install ultralytics opencv-python huggingface-hub

Usage:
    python face_detection_yolov8_hf.py --source <path_or_camera> [--conf 0.25] [--iou 0.45] [--no-view] [--save]

Examples:
    python face_detection_yolov8_hf.py --source test.jpg --save
    python face_detection_yolov8_hf.py --source 0 --no-view
"""

import os
import argparse
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2
from util.VideoProcessor import VideoProcessor

def download_weights(repo_id: str, filename: str, cache_dir: str = 'weights') -> str:
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, filename)
    if not os.path.isfile(local_path):
        print(f"Downloading weights '{filename}' from Hugging Face repository '{repo_id}'...")
        local_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
        print(f"Saved to: {local_path}")
    return local_path

def detect_faces(
    source: str,
    model_weights: str,
    conf: float = 0.25,
    iou: float = 0.45,
    view: bool = True,
    save: bool = False,
    save_dir: str = 'output'):
    # Load YOLOv8 model
    model = YOLO(model_weights)

    # Run inference
    results = model.predict(source=source, conf=conf, iou=iou)

    for r in results:
        img = r.orig_img.copy()
        boxes = r.boxes.xyxy.cpu().numpy()
        confidences = r.boxes.conf.cpu().numpy()

        for (x1, y1, x2, y2), c in zip(boxes, confidences):
            if c < conf:
                continue
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, f'{c:.2f}', (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if view:
            cv2.imshow('YOLOv8-Face', img)
            cv2.waitKey(1)

        if save:
            os.makedirs(save_dir, exist_ok=True)
            fname = os.path.basename(source) if not source.isnumeric() else 'webcam'
            out_path = os.path.join(save_dir, f'detected_{fname}')
            cv2.imwrite(out_path, img)
            print(f'Saved: {out_path}')

    if view:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv8 Face Detection (Hugging Face)')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to image/video or camera index (e.g., 0)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='NMS IoU threshold')
    parser.add_argument('--no-view', action='store_true',
                        help='Disable display of detections')
    parser.add_argument('--save', action='store_true',
                        help='Save detection results to `output/`')
    args = parser.parse_args()

    # Download pretrained weights from Hugging Face
    # model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")

    # repo_id = 'arnabdhar/YOLOv8-Face-Detection'
    # filename = 'model.pt'
    # weights_path = download_weights(repo_id, filename)

    # Run face detection
    # detect_faces(
    #     source=args.source,
    #     model_weights=weights_path,
    #     conf=args.conf,
    #     iou=args.iou,
    #     view=not args.no_view,
    #     save=args.save
    # )
    
    processor = VideoProcessor(
        model_path="yolov8n-face-lindevs.pt",
        input_video_path=args.source,
    )

    processor.process_video()
