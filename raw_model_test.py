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
model = YOLO("custom_model.pt", task="detect")

# object classes
classNames = ["correct", "error", "not-lid"]

print("TEST START")
img = cv2.imread("test_photo.jpeg")

start =  time.time_ns() / (10 ** 9)
results = model.predict(img, conf=0.5)
end = time.time_ns() / (10 ** 9) 
print("total time to detect: ")
print((end-start)*1000)
print("ms")


start =  time.time_ns() / (10 ** 9)
results = model.predict(img, conf=0.5, stream = True)
end = time.time_ns() / (10 ** 9) 
print("total time to detect: ")
print((end-start)*1000)
print("ms")

for r in results:
    boxes = r.boxes
    for box in boxes:
        # confidence
        confidence = math.ceil((box.conf[0]*100))/100
        print("Confidence --->",confidence)

        # class name
        cls = int(box.cls[0])
        print("Class name -->", classNames[cls])
