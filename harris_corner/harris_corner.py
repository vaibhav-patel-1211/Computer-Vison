# import
from concurrent.futures import thread
import trace
import numpy as np
import cv2 as cv

# reading image
img = cv.imread('Assets/harris_corner.png')

if img is None :
  print("Image not found")
else :

  # convert image into gray scale
  gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

  # harris need float value
  gray = np.float32(gray)

  # computer image gradient
  Ix = cv.Sobel(gray,cv.CV_32F,1,0,ksize=3)
  Iy = cv.Sobel(gray,cv.CV_32F,0,1,ksize=3)

  # computer harris matrix
  Ixx = Ix * Ix
  Ixy = Ix * Iy
  Iyy = Iy * Iy

  # apply gaussian blur to the harris matrix
  Sxx = cv.GaussianBlur(Ixx,(3,3),sigmaX=1)
  Syy = cv.GaussianBlur(Iyy,(3,3),sigmaX=1)
  Sxy = cv.GaussianBlur(Ixy,(3,3),sigmaX=1)

  # compute harris corner response R
  k = 0.04
  det_M = (Sxx * Syy) - (Sxy * Sxy)
  trace_M = Sxx + Syy

  R = det_M - k * (trace_M ** 2)

  # threshold on R to find corner
  threshold = 0.01 * R.max()

  corner_img = img.copy()
  corner_img[R>threshold] = [0,0,255]

  cv.imshow('Output',corner_img)
  cv.waitKey(0)
  cv.destroyAllWindows()
