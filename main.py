import numpy as np
import cv2

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # cv2.rectangle(gray, (x, y), (x+w, y+h), (255, 0, 0), 1)
        ret_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(ret_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.circle(ret_gray, (ex+ew//2, ey+eh//2), 20, (255, 0, 0), 1)
            params = cv2.SimpleBlobDetector_Params()
            eye = ret_gray[y+ey:y+h+ey+eh, x+ex:x+w+ex+ew]
            params.filterByArea = True
            params.minArea = 0
            params.maxArea = 4000
            detector = cv2.SimpleBlobDetector_create(params)
            blobs = detector.detect(eye)
            if blobs:
                print("detected")
                keypoint = blobs[0]
                x = int(keypoint.pt[0])
                y = int(keypoint.pt[1])
                s = keypoint.size
                r = int(s//2)
                cv2.circle(gray, (x, y), r, (255, 255, 255), 2)
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
