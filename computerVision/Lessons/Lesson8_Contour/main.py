import cv2 as cv

img = cv.imread(r"C:\Users\user\Desktop\deneme\computerVision\Lessons\Lesson8_Contour\birds.jpg")

# 1. BGR to GrayScale
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 2. Thresholding - GrayScale to Binary Image
ret, thresh = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY_INV)

# 3. Find Contours
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# 4. Draw Contours
for cnt in contours:
    if cv.contourArea(cnt) > 200:
        # cv.drawContours(img, cnt, -1, (0, 255, 0), 1)
        x1, y1, w, h = cv.boundingRect(cnt)
        cv.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

# Above code is a basic object detection code

cv.imshow("image", img)
cv.imshow("thresh", thresh)
cv.waitKey(0)
