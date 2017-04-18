# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
from grip import GripPipeline
import numpy as np

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

gripped = GripPipeline()

# allow the camera to warmup
time.sleep(0.1)
font = cv2.FONT_HERSHEY_SIMPLEX

def validate(c):
    if c[2] <= -45:
        return (c[1][1], c[1][0], c[2])
    return (c[1][0], c[1][1], c[2])

def put_text(c, x):
    
    cv2.putText(image, "Width:{}".format (round(c[0], 3)) , (x , 12), font, .5 , (255,255,255), 1)
    cv2.putText(image, "Height:{}".format (round(c[1], 3)) , (x , 24), font, .5, (255,255,255), 1)
    cv2.putText(image, "Angle:{}".format (round(c[2]), 3) , (x , 36), font, .5 , (255,255,255), 1)

def draw(c, c1):
    x3 = draw_line_centre(c, c1)
    
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(image,[box],0,(0,0,255),2)

    c = validate(rect)

    rect = cv2.minAreaRect(c1)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(image,[box],0,(0,0,255),2)

    c1 = validate(rect)

    cv2.putText(image, "Center X:{}".format (x3) , (10 , 470), font, .5 , (255,255,255), 1)


    put_text(c1, 10 )
    put_text(c, x3 + 10)

def draw_line_centre(c1, c2):
    m1 = cv2.moments(c1)
    m2 = cv2.moments(c2)

    x1= int(m1["m10"] / m1["m00"])
    y1 = int(m1["m01"] / m1["m00"])

    x2 = int(m2["m10"] / m2["m00"])
    y2 = int(m2["m01"] / m2["m00"])

    x3 = int((x1 + x2) / 2)

    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0))
    cv2.line(image, (x3, 0), (x3, 480), (0, 255, 0))

    return x3

#i =0

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array
    gripped.process(image)

    #if i % 100 == 0:
    #    cv2.imwrite("image{}{}".format(i, ".jpeg"), image)
    #i = i +1

    cv2.drawContours(image, gripped.filter_contours_output, -1 , (255, 0, 0), 3)

    if(len(gripped.filter_contours_output) == 2):
        c = gripped.filter_contours_output
        draw(c[0], c[1])

    # show the frame
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
cv2.destroyAllWindows()
