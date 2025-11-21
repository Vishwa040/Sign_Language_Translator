import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

folder = "Data/g"
if not os.path.exists(folder):
    os.makedirs(folder)

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame. Exiting...")
            break
        
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            
            x1, y1 = max(0, x - offset), max(0, y - offset)
            x2, y2 = min(img.shape[1], x + w + offset), min(img.shape[0], y + h + offset)
            imgCrop = img[y1:y2, x1:x2]
            
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCropShape = imgCrop.shape
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
            
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)
        
        # Display the main image
        cv2.imshow("Image", img)
        
        key = cv2.waitKey(1)
        if key == ord("s"):
            counter += 1
            imgWhiteWithCounter = imgWhite.copy()
            text = f"Image_{counter}"
            cv2.putText(imgWhiteWithCounter, text, (10, imgSize - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            filename = f'{folder}/Image_{counter}.jpg'
            cv2.imwrite(filename, imgWhiteWithCounter)
            print(f"Counter: {counter}")  
        elif key == 27:  
            print("Exiting...")
            break

finally:
    cap.release()
    cv2.destroyAllWindows()