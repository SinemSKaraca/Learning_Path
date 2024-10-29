import os
import cv2 as cv

img = cv.imread(os.path.abspath("Lesson4_Blurring/cow-salt-peper.png"))

k_size = 7     # kernel size -> the larger this number the more intense the blur

# AVERAGING
img_blur = cv.blur(img, (k_size, k_size))

# GAUSSIAN
img_gaussian = cv.GaussianBlur(img, (k_size, k_size), 3)

# MEDIAN
img_median = cv.medianBlur(img, k_size)

cv.imshow("Image", img)
cv.imshow("Blurred Image", img_blur)
cv.imshow("Gaussian Blurred Image", img_gaussian)
cv.imshow("Median Blurred Image", img_median)
cv.waitKey(0)