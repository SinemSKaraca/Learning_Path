import os
import cv2 as cv

img = cv.imread(os.path.abspath("Lesson5_Threshold/bear.jpg"))

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# GLOBAL THRESHOLD
ret, thresh = cv.threshold(img_gray, 80, 255, cv.THRESH_BINARY)   # pixel < 80 - black, pixel > 80 - white

# Let's reduce the noice on the background
thresh = cv.blur(thresh, (10, 10))
ret, thresh = cv.threshold(thresh, 80, 255, cv.THRESH_BINARY)

cv.imshow("img", img)
cv.imshow("thresh", thresh)
cv.waitKey(0)

