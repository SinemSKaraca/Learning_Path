import os
import cv2 as cv

img = cv.imread("Lesson2_BasicOperations/cat.jpg")
print("Original..: ", img.shape)

# resizing
resized_img = cv.resize(img, (320, 213))   # resize --> (width, height)
print("Resized..: ", resized_img.shape)          # shape --> (height, width, channels)

cv.imshow("CAT", img)
cv.imshow("RESIZED CAT", resized_img)
cv.waitKey(0)
