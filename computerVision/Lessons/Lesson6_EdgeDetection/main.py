import os
import cv2 as cv
import numpy as np


imgPath = r"C:\Users\user\Desktop\deneme\computerVision\Lessons\Lesson6_EdgeDetection\player.jpg"
img = cv.imread(imgPath)

img_edge = cv.Canny(img, 100, 200) # (image, minThreshold, maxThreshold)

# DILATE - Dilation is a process that grows or thickens the white regions (foreground) in a binary image.
img_edge_dilate = cv.dilate(img_edge, np.ones((5, 5), dtype=np.int8))

# ERODE - It shrinks or thins the white regions (foreground) in a binary image.
img_edge_erode = cv.erode(img_edge_dilate, np.ones((3, 3), dtype=np.int8))


cv.imshow("img", img)
cv.imshow("img_edge", img_edge)
cv.imshow("img_edge_dilate", img_edge_dilate)
cv.imshow("img_edge_erode", img_edge_erode)
cv.waitKey(0)





