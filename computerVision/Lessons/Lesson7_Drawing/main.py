import cv2 as cv

img = cv.imread(r"C:\Users\user\Desktop\deneme\computerVision\Lessons\Lesson7_Drawing\whiteboard.png")

# LINE
cv.line(img, (100, 150), (300, 450), (0, 255,0), 3)
'''(image, (starting point), (ending point), (color), thickness)'''

# RECTANGLE
cv.rectangle(img, (200, 350), (450, 600), (0, 0, 255), -1)   # -1 -> Fills the rectangle with selected color
'''(image, (sol üst köşe), (sağ alt köşe), (color), thickness)'''

# CIRCLE
cv.circle(img, (800, 200), 75, (255, 0, 0), 10)

# TEXT
cv.putText(img, "Hey you!", (200, 450), cv.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 255), 2)

cv.imshow("image", img)
cv.waitKey(0)