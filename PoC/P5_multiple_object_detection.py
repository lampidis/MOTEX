import atexit
import time
import cv2
import serial
import numpy as np
import torch
from ultralytics import YOLO
import supervision as sv

#variables
frame_skip_interval = 3  # ✅ Process YOLO every 3 frames
frame_counter = 0

# ✅ Load Camera Calibration Data
calibration_data = np.load("camera_calibration_data.npz")
camera_matrix = calibration_data["camera_matrix"]
dist_coeffs = calibration_data["dist_coeffs"]

# ✅ Initialize YOLOv8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on {DEVICE.upper()}")
model = YOLO("yolov8n.pt").to("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Open Helmet Camera
camera_index = 0
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

# ✅ Serial connection to Arduino
arduino = serial.Serial("COM8", 9600, timeout=1)  
time.sleep(2)  # ✅ Allow time for Arduino to initialize
arduino.write("RESET\n".encode())  # ✅ Reset LEDs at startup

# ✅ Helmet & LED Configuration
NUM_LEDS = 69
FRAME_WIDTH = 1280
FOV_HORIZONTAL = 80  # Degrees

# ✅ Predefined colors (R,G,B) for multiple cars
colors = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255)   # Cyan
]

def map_position_to_led(center_x):
    """ Maps detected car position to an LED index with high accuracy. """
    
    # ✅ Convert pixel position to angle (degrees)
    theta = ((center_x - FRAME_WIDTH / 2) / (FRAME_WIDTH / 2)) * (FOV_HORIZONTAL / 2)

    # ✅ Direct mapping (No smoothing)
    led_index = int(((theta + (FOV_HORIZONTAL / 2)) / FOV_HORIZONTAL) * NUM_LEDS)

    # ✅ Flip LED index if necessary
    led_index = NUM_LEDS - 1 - led_index  

    # ✅ Ensure within valid LED range
    led_index = max(0, min(NUM_LEDS - 1, led_index))
    
    print(f"🎯 Mapped Center X: {center_x} → θ: {theta:.2f}° → LED: {led_index}")
    
    return led_index

last_sent_led_data = ""  # ✅ Initialize variable to store last sent LED data

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame.")
        break

    frame_counter += 1

    # ✅ Skip frames to reduce YOLO computation load
    if frame_counter % frame_skip_interval == 0:
        #gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = model(frame, verbose=False, conf=0.3, iou=0.4)[0]  

        detections = sv.Detections.from_ultralytics(results)

        detected_led_data = []  

        for i, (det, label) in enumerate(zip(detections.xyxy, detections.class_id)):
            class_name = model.names[label]
            if class_name == "car":
                x1, y1, x2, y2 = map(int, det)
                center_x = (x1 + x2) // 2
                led_index = map_position_to_led(center_x)

                # ✅ Assign a color (loop through predefined colors)
                color = colors[i % len(colors)]  
                detected_led_data.append(f"{led_index}:{color[0]},{color[1]},{color[2]}")  

        if detected_led_data:
            led_data = ";".join(detected_led_data)
            
            # ✅ Check if last sent data is different (Avoid sending redundant data)
            if led_data != last_sent_led_data:
                print(f"🔴 Sending Multiple LED Colors: {led_data}")
                arduino.write(f"{led_data}\n".encode())  
                last_sent_led_data = led_data  # ✅ Store last sent message


    cv2.imshow("YOLOv8 Helmet Car Detection & LED Mapping", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ✅ Cleanup when script exits
cap.release()
cv2.destroyAllWindows()
arduino.write("OFF\n".encode())  # ✅ Turn off LEDs before exiting
