from ultralytics import YOLO
import cv2
import math 
import time
from picamera2 import Picamera2

from PIL import Image
from libcamera import controls

picam2 = Picamera2()
picam2.set_controls({"FrameRate": 60})  # 30 FPS olarak ayarlan?r


lens_position = 5.5
picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": lens_position})

# model
model = YOLO("custom_model.pt")

# object classes
classNames = ["correct", "error", "not-lid"]

print("TEST START")
img = cv2.imread("test_photo3.jpeg")
cv2.imshow("Frame new", img)
#to increase speed
results = model(img, stream=True)

start = time.time()
results = model(img, stream=True)
end = time.time()
print("total time to detect: ")
print((end-start)*1000)
print("ms")



picam2.start()
counter = 0;
while True:
    image = picam2.capture_array()
    cv2.imshow("Frame", image)
    if(cv2.waitKey(1) == ord("q")):
        break
    if(cv2.waitKey(1) == ord("b")):
        print("B basildi.")
        if image.shape[2] == 4: 
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)q
        #cv2.imshow("Frame new", image)
        
        start = time.time()
        results = model(image, stream=True)
        end = time.time()
        print("total time to detect: ")
        print((end-start)*1000)
        print("ms")

        for r in results:
            if r is None or len(r) == 0:
                print("error")
                continue
            print("res")
            boxes = r.boxes
            if not boxes:
                print("boxes eerror")
                continue
                
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # put box in cam
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                print("Confidence --->",confidence)

                # class name
                cls = int(box.cls[0])
                print("Class name -->", classNames[cls])

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(image, classNames[cls], org, font, fontScale, color, thickness)
                cv2.imwrite("train"+str(counter) + ".png", image)
                counter = counter + 1;
cv2.destroyAllWindows()

