import cv2 as cv

# READ WEBCAM
webcam = cv.VideoCapture(0)


# VISUALIZE WEBCAM
while True:
    ret, frame = webcam.read()
    cv.imshow("Frame", frame)
    if cv.waitKey(40) & 0xFF == ord('q'):
        break


webcam.release()
cv.destroyAllWindows()