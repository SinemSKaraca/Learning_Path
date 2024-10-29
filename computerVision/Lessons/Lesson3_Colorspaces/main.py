import os
import cv2 as cv

# This picture is in BGR colorspace (Blue-Green-Red)

img = cv.imread(os.path.abspath("Lesson3_Colorspaces/bird.jpg"))

# CONVERT COLORSPACE
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)              # BGR to RGB
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)            # BGR to Gray Scale
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)              # BGR to HSV

cv.imshow("BGR", img)
cv.imshow("RGB", img_rgb)
cv.waitKey(0)

'''cv.imshow("Bird", img)
cv.imshow("Gray Scale", img_gray)
cv.waitKey(0)'''

'''cv.imshow("BGR", img)
cv.imshow("HSV", img_hsv)
cv.waitKey(0)'''


