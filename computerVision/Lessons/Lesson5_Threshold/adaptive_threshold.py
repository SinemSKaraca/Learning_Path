import cv2 as cv

img = cv.imread("Lesson5_Threshold/handwritten.png")
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# GLOBAL THRESHOLD
ret, global_thresh = cv.threshold(img_gray, 60, 255, cv.THRESH_BINARY) # (ret = thresehold value)

# ADAPTIVE THRESHOLD
adaptive_thresh = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 30)

cv.imshow("img", img)
cv.imshow("global threshold", global_thresh)
cv.imshow("adaptive threshold", adaptive_thresh)
cv.waitKey(0)
