import cv2
import numpy as np

class CrashDetector:
    def __init__(self, fps=30):
        self.fps = fps
        self.tracked_objects = {}
        self.prev_speeds = {}
        self.collisions = {}
        self.frame_interval = 5
        self.T = self.frame_interval / fps

    def update_tracks(self, detections):
        updated_tracks = {}

        for tracker_id, (x1, y1, x2, y2) in zip(detections.tracker_id, detections.xyxy):
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            width, height = abs(x2 - x1), abs(y2 - y1)
            bbox = (x1, y1, x2, y2)
            updated_tracks[tracker_id] = {'centroid': (cx, cy), 'bbox': bbox, 'size': (width, height)}

        return updated_tracks

    def detect_collisions(self, frame_idx):
        collision_pairs = []
        for obj1, data1 in self.tracked_objects.items():
            c1, bbox1 = data1['centroid'], data1['bbox']
            for obj2, data2 in self.tracked_objects.items():
                if obj1 >= obj2:
                    continue  # Avoid duplicate checks
                c2, bbox2 = data2['centroid'], data2['bbox']

                # Bounding Box Overlap Check
                if (2 * abs(c1[0] - c2[0]) < (data1['size'][0] + data2['size'][0])) and \
                   (2 * abs(c1[1] - c2[1]) < (data1['size'][1] + data2['size'][1])):
                    collision_pairs.append((obj1, obj2))
                    if (obj1, obj2) not in self.collisions:
                        self.collisions[(obj1, obj2)] = []
                    self.collisions[(obj1, obj2)].append(frame_idx)
        return collision_pairs