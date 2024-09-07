from ultralytics import YOLO
import cv2
import math 
import time

from PIL import Image

# model
model = YOLO("custom_model.pt")

# object classes
classNames = ["correct", "error", "not-lid"]

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

#picam2 = Picamera2()
#picam2.start()
counter = 0;
while True:
    #image = picam2.capture_array()
    success, image = cap.read()
    cv2.imshow("Frame", image)
    if(cv2.waitKey(1) == ord("q")):
        break
    if(cv2.waitKey(1) == ord("b")):
        #cv2.imwrite("train"+str(counter) + ".png", image)
        start = time.time()
        results = model(image, stream=True)
        end = time.time()
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
        counter = counter + 1
cv2.destroyAllWindows()


#image = cv2.imread("test_photo3.jpeg")

