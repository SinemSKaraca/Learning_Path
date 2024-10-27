import os
import cv2 as cv

# READ IMAGE
image_path = os.path.abspath("./Lesson1_IO/data/bird.jpg")
img = cv.imread(image_path)


# WRITE IMAGE
cv.imwrite(os.path.abspath("./Lesson1_IO/data/bird_out.jpg"), img)


# VISUALIZE IMAGE
cv.imshow('Image', img)
cv.waitKey(0)   # If we don't add this, window gets opened and immediately after closed. We can't see.