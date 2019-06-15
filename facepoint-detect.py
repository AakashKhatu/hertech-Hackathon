from imutils import face_utils
import dlib
import cv2
import json
#
p = "shape_predictor_68_face_landmarks.dat"
# default model of dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)
 
while True:
    _, image = cap.read()
    #not checking if webcam is available as it is guaranteed . replace this with 
    # if else to check camera access.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 0), -1)
        print(shape)
    cv2.imshow("Output", image)
    k = cv2.waitKey(5) & 0xFF #q
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
