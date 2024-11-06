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

#TASK 3

def erode(image, kernel):
    m, n = image.shape
    km, kn = kernel.shape
    hkm = km // 2
    hkn = kn // 2
    eroded = np.copy(image)

    for i in range(hkm, m - hkm):
        for j in range(hkn, n - hkn):
            eroded[i, j] = np.min(
            image[i - hkm:i + hkm + 1, j - hkn:j + hkn + 1][kernel == 1]
            )

def dilate(image, kernel):
    m, n = image.shape
    km, kn = kernel.shape
    hkm = km // 2
    hkn = kn //2
    dilated = np.copy(image)

    for i in range(hkm, m - hkm):
        for j in range(hkn, n - hkn):
            dilated[i, j] = np.max(
                image[i - hkm:i + hkm + 1, j - hkn:j + hkn + 1][kernel == 1]
            )
def morphological_transform():
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

        kernel = np.ones((5, 5), np.uint8)

        open_img = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        close_img = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


        cv2.imshow('open_frame', open_img)
        cv2.imshow('close_frame', close_img)

        if cv2.waitKey(30) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

#TASK 4
def capture_in_contour():
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

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 0:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

        cv2.imshow('red_frame', frame)

        if cv2.waitKey(30) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

morphological_transform()