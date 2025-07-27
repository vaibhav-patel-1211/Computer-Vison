from concurrent.futures import thread
import cv2 as cv
from networkx import center, radius
import numpy as np
import matplotlib.pyplot as plt

# load image
img = cv.imread('Assets/blob-detection.png')

if img is None :
  print('Image not found')
else :
  # convert to gray scale image
  gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

  # apply gaussian blur
  blurred = cv.GaussianBlur(gray,(7,7),2)

  # apply laplacian filter
  laplacian = cv.Laplacian(blurred,cv.CV_64F)
  laplacian = np.absolute(laplacian)
  laplacian = np.uint8(laplacian)

  # threshold to get string blob
  ret,threshold = cv.threshold(laplacian,10,255,cv.THRESH_BINARY)

  # find countours - outline possible blob
  contours, _ = cv.findContours(threshold,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


  # draw circle arround detected blob
  blob = 0
  for cnt in contours :
    (x,y), radius = cv.minEnclosingCircle(cnt)
    if radius > 5 : # filter out noise
      center = (int(x),int(y))
      cv.circle(img,center,int(radius),(0,0,255),2)
      blob +=1
  # Show results using OpenCV windows
  cv.imshow('Original', gray)
  cv.imshow('Threshold', threshold)
  cv.imshow('Detected Blobs', img)

  print(f"Number of blobs detected: {blob}")
  print("Press any key to close windows")
  cv.waitKey(0)
  cv.destroyAllWindows()

