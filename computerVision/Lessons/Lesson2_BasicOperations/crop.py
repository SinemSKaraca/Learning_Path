import os
import cv2 as cv

img = cv.imread(os.path.abspath("Lesson2_BasicOperations/cat.jpg"))
print(img.shape)

# We will crop the image so we only get the cat in the image
cropped_img = img[61:427, 320:640]

cv.imshow("Image", img)
cv.imshow("Cropped Image", cropped_img)
cv.waitKey(0)



