import cv2

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

readIPWriteTOFile()
