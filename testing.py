import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.utils import get_custom_objects

# Custom Depthwise Convolution Layer
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')
        super(CustomDepthwiseConv2D, self).__init__(*args, **kwargs)

get_custom_objects().update({'DepthwiseConv2D': CustomDepthwiseConv2D})

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300
labels = ["H", "E" , "L", "O", "A", "5", "6"]

last_prediction = None
start_time = None
hold_time_threshold = 3  
recognized_text = []

frame_count = 0
total_processing_time = 0
correct_predictions = 0
total_predictions = 0
true_labels = ["H", "E", "L", "O", "A", "5", "6"] 

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    frame_start_time = time.time()  

    imgOutput = img.copy()

    cv2.rectangle(imgOutput, (0, 0), (imgOutput.shape[1], 100), (255, 255, 255), -1)
    cv2.putText(imgOutput, ''.join(recognized_text), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2,
                (0, 0, 0), 3)

    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        if w <= 0 or h <= 0 or x < 0 or y < 0:
            print("Invalid hand bounding box dimensions")
            continue

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        
        y_start = max(y - offset, 0)
        y_end = min(y + h + offset, img.shape[0])
        x_start = max(x - offset, 0)
        x_end = min(x + w + offset, img.shape[1])

        imgCrop = img[y_start:y_end, x_start:x_end]

        if imgCrop.size == 0:
            print("Cropped image is empty")
            continue
            
        aspectRatio = h / w
        
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
        
        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        current_label = labels[index]
        
        total_predictions += 1
        if current_label in true_labels:
            correct_predictions += 1

        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, current_label, (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7,
                    (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

        if current_label == last_prediction:
            if start_time and (time.time() - start_time >= hold_time_threshold):
                recognized_text.append(current_label)
                start_time = None  
        else:
            last_prediction = current_label
            start_time = time.time()  

    processing_time = time.time() - frame_start_time
    total_processing_time += processing_time
    frame_count += 1

    fps = frame_count / total_processing_time if total_processing_time > 0 else 0
    cv2.putText(imgOutput, f'FPS: {fps:.2f}', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Image", imgOutput)
    
    if cv2.waitKey(1) >= 0:
        break

accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
average_processing_time = total_processing_time / frame_count if frame_count > 0 else 0

print(f"Frame Rate (FPS): {fps:.2f}")
print(f"Detection Accuracy: {accuracy:.2f}%")
print(f"Average Processing Time per Frame: {average_processing_time:.4f} seconds")

cap.release()
cv2.destroyAllWindows()
