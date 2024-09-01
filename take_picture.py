import cv2
from picamera2 import Picamera2

picam2 = Picamera2()
picam2.start()
counter = 0;
while True:
    image = picam2.capture_array()
    cv2.imshow("Frame", image)
    if(cv2.waitKey(1) == ord("q")):
        break
    if(cv2.waitKey(1) == ord("b")):
        cv2.imwrite("train"+str(counter) + ".png", image)
        counter++
cv2.destroyAllWindows()
