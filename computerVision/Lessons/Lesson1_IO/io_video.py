import os
import cv2 as cv

# READ VIDEO
video_path = os.path.abspath("Lesson1_IO/data/dog.mp4")
video = cv.VideoCapture(video_path)


# VISUALIZE VIDEO
ret = True

while ret:
    ret, frame = video.read()

    if ret:
        cv.imshow("Frame", frame)
        cv.waitKey(40)

video.release()
cv.destroyAllWindows()
