import numpy as np
import cv2
import sys
import os


cascade = cv2.CascadeClassifier(os.path.abspath('./cascade/haarcascade_frontalface_default.xml'))

# import image
img = cv2.imread(sys.argv[1])

# convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = cascade.detectMultiScale(gray)
  
if len(faces) == 0:
    print("no faces")
  
else:
    print("num faces: " + str(faces.shape[0]))