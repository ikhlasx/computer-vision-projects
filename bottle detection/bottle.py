import cv2 as cv
import cvzone
from ultralytics import YOLO
import math
from sort import *
import numpy as np

cap = cv.VideoCapture('C:/Users/Ikhlas/Desktop/ikhlas/Robotics and AI/projects/Pro all detect/YOUTUBE-TUTORIAL-CODES-main/bottle detection/bot1.mp4')
model = YOLO('C:/Users/Ikhlas/Desktop/ikhlas/Robotics and AI/projects/Impex/Yolo_plate_detector/weights/runs/detect/train/weights/best.pt')

# Initialize tracker
tracker = Sort(max_age=20, min_hits=3)

# Line position for detection (adjust according to conveyor belt setup)
line = [1100, 0, 1100, 900]

# List to store detected plate IDs
counterin = []

# Load class names
classnames = []
with open('path_to_your_classes.txt', 'r') as f:
    classnames = f.read().splitlines()

# Main loop to process each frame
while True:
    ret, img = cap.read()

    # Restart the video if it ends
    if not ret:
        cap = cv.VideoCapture('path_to_your_plate_video.mp4')
        continue

    detections = np.empty((0, 5))

    # Run YOLO model to detect objects in the frame
    results = model(img, stream=True)
    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            confidence = box.conf[0]
            class_detect = box.cls[0]
            class_detect = int(class_detect)
            class_detect = classnames[class_detect]
            conf = math.ceil(confidence * 100)

            # Detect only plates with high confidence
            if class_detect == 'plate' and conf >= 80:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Store current detection coordinates and confidence
                current_detections = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_detections))

    # Update tracker with current detections
    tracker_result = tracker.update(detections)

    # Draw the detection line
    cv.line(img, (line[0], line[1]), (line[2], line[3]), (0, 255, 255), 5)

    # Process tracking results
    for track_result in tracker_result:
        x1, y1, x2, y2, id = track_result
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        w, h = x2 - x1, y2 - y1  # Plate dimensions (w: length of the plate)
        cx, cy = x1 + w // 2, y1 + h // 2  # Center of the plate

        # Draw a rectangle around detected plates and display the ID
        cvzone.cornerRect(img, [x1, y1, w, h], rt=5)
        cvzone.putTextRect(img, f"{id}", [x1 + 8, y1 - 12], scale=2, thickness=2)

        # Detect if the plate crosses the line
        if line[1] < cy < line[3] and line[2] - 10 < cx < line[2] + 10:
            cv.line(img, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 10)
            if counterin.count(id) == 0:  # Count unique plate ID
                counterin.append(id)

        # Display the length (width) of the plate
        cvzone.putTextRect(img, f"Length: {w} pixels", [x1, y2 + 20], scale=1.5, thickness=2)

    # Display total count of plates
    cvzone.putTextRect(img, f'Total Plates = {len(counterin)}', [500, 34], thickness=4, scale=2.3, border=2)

    # Show the frame
    cv.imshow('Plate Detection', img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
