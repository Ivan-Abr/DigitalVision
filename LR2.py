import cv2
import numpy as np


#TASK 1
def video_hsv():
    cap = cv2.VideoCapture("http://192.168.0.104:8080/video")
    if not cap.isOpened():
        print("Невозможно открыть файл")
        exit()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.imshow('hsv_frame', hsv)

        if cv2.waitKey(30) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()



#TASK 2
def video_hsv_red():
    cap = cv2.VideoCapture("http://192.168.0.104:8080/video")
    if not cap.isOpened():
        print("Невозможно открыть файл")
        exit()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        low_red = np.array([0, 120, 220])
        high_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, low_red, high_red)
        red_frame = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow('red_frame', red_frame)

        if cv2.waitKey(30) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

video_hsv_red()