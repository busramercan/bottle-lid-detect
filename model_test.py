from ultralytics import YOLO
import cv2
import math 
import time
from picamera2 import Picamera2

from PIL import Image
from libcamera import controls

picam2 = Picamera2()
picam2.set_controls({"FrameRate": 60})  # 30 FPS olarak ayarlan?r


lens_position =5.5
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
    counter = counter + 1
    image = picam2.capture_array()
    cv2.imshow("Frame", image)
    if(cv2.waitKey(1) == ord("q")):
        break
    if(counter > 50):
        print("B basildi.")
        if image.shape[2] == 4: 
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)q
        cv2.imshow("Frame new", image)
        #cv2.imwrite("train"+str(counter) + ".png", image)
        start = time.time()
        results = model(image, stream=True)
        end = time.time()
        time.sleep(10)
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
                # confidence
                print("conf")
                confidence = math.ceil((box.conf[0]*100))/100
                print("Confidence --->",confidence)

                # class name
                cls = int(box.cls[0])
                print("Class name -->", classNames[cls])
        
cv2.destroyAllWindows()

