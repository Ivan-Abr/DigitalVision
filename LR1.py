import cv2
import numpy as np

#TASK 2
def read_image():
    extensions = ['jpg','png','bmp']
    read_flags = [cv2.IMREAD_GRAYSCALE, cv2.IMREAD_COLOR, cv2.IMREAD_UNCHANGED]
    window_flags = [cv2.WINDOW_NORMAL, cv2.WINDOW_FULLSCREEN, cv2.WINDOW_AUTOSIZE]
    for ext in extensions:
        filename = r'E:\python\visionLabs\data\img.' + ext
        for read_flag in read_flags:
            img = cv2.imread(filename, read_flag)
            for window_flag in window_flags:
                win_name = f'window: img.{ext}, {read_flag}, {window_flag}'
                print(win_name)
                cv2.namedWindow(win_name, window_flag)
                cv2.imshow(win_name, img)
                cv2.waitKey(0)
    cv2.destroyAllWindows()

#TASK 3
def read_video():
    cap = cv2.VideoCapture('data/helloArbuz.mp4')
    if not cap.isOpened():
        print("Невозможно открыть файл")
        exit()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (360, 360))
        filter_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Original', frame)
        cv2.imshow('Resized', resized_frame)
        cv2.imshow('Filter', filter_frame)
        if cv2.waitKey(30) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

#TASK 4
def readIPWriteTOFile():
    video = cv2.VideoCapture(r'data/helloArbuz.mp4', cv2.CAP_ANY)
    ok, vid = video.read()

    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter("data/byebyeArbuz.mp4", fourcc, 25, (w, h))

    while True:
        ok, vid = video.read()

        cv2.imshow('Video', vid)
        video_writer.write(vid)

        if cv2.waitKey(30) & 0xFF == 27:
            break

    video.release()
    video_writer.release()
    cv2.destroyAllWindows()

#TASK 5
def to_HSV_format():
    img = cv2.imread(r"E:\python\visionLabs\data\img.png")
    img_edit = cv2.imread(r"E:\python\visionLabs\data\img.png")

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('edited image', cv2.WINDOW_NORMAL)

    cv2.imshow('image', img)

    hsv = cv2.cvtColor(img_edit, cv2.COLOR_BGR2HSV)
    cv2.imshow('edited image', hsv)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

#TASK 6
def read_camera():
    cap = cv2.VideoCapture("http://192.168.0.103:8080/video")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        height, width, _ = frame.shape
        cross_image = np.zeros((height, width, 3), dtype=np.uint8)
        vertical_line_width = 60
        vertical_line_height = 300
        cv2.rectangle(
            cross_image,
            (width // 2 - vertical_line_width // 2, height // 2 - vertical_line_height // 2),
            (width // 2 + vertical_line_width // 2, height // 2 + vertical_line_height // 2),
            (0, 0, 255),
            2
        )
        horizontal_line_width = 250
        horizontal_line_height = 55
        cv2.rectangle(
            cross_image,
            (width // 2 - horizontal_line_width // 2, height // 2 - horizontal_line_height // 2),
            (width // 2 + horizontal_line_width // 2, height // 2 + horizontal_line_height // 2),
            (0, 0, 255),
            2
        )
        result_frame = cv2.addWeighted(frame, 1, cross_image, 0.5, 0)
        cv2.imshow("Red Cross", result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

#TASK 7
def readCameraWriteTOFile():
    video = cv2.VideoCapture("http://192.168.0.103:8080/video", cv2.CAP_ANY)
    ok, vid = video.read()

    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter("data/camera.mp4", fourcc, 25, (w, h))

    while True:
        ok, vid = video.read()

        cv2.imshow('Video', vid)
        video_writer.write(vid)

        if cv2.waitKey(30) & 0xFF == 27:
            break

    video.release()
    video_writer.release()
    cv2.destroyAllWindows()

#TASK 8
def fillCross():
    cap = cv2.VideoCapture("http://192.168.0.103:8080/video")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        height, width, _ = frame.shape
        cross_image = np.zeros((height, width, 3), dtype=np.uint8)
        vertical_line_width = 60
        vertical_line_height = 300
        cv2.rectangle(
            cross_image,
            (width // 2 - vertical_line_width // 2, height // 2 - vertical_line_height // 2),
            (width // 2 + vertical_line_width // 2, height // 2 + vertical_line_height // 2),
            (0, 255, 255),
            -1
        )
        horizontal_line_width = 250
        horizontal_line_height = 55
        cv2.rectangle(
            cross_image,
            (width // 2 - horizontal_line_width // 2, height // 2 - horizontal_line_height // 2),
            (width // 2 + horizontal_line_width // 2, height // 2 + horizontal_line_height // 2),
            (0, 255, 255),
            -1
        )
        result_frame = cv2.addWeighted(frame, 1, cross_image, 0.5, 0)
        cv2.imshow("Red Cross", result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
fillCross()
