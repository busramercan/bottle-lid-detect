from ultralytics import YOLO
import cv2
import math 
import time

from PIL import Image
# start webcam
# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
#cap.set(4, 480)

# model
model = YOLO("custom_model.pt")

# object classes
classNames = ["correct", "error", "not-lid"]

print("TEST START")

start = time.time()
img = cv2.imread("test_photo3.jpeg")
end = time.time()
print("total time to detect: ")
print((end-start)*1000)
print("ms")

results = model(img, stream=True)
for r in results:
    boxes = r.boxes
    for box in boxes:
        # confidence
        confidence = math.ceil((box.conf[0]*100))/100
        print("Confidence --->",confidence)

        # class name
        cls = int(box.cls[0])
        print("Class name -->", classNames[cls])

