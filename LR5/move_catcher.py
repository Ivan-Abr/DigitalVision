import cv2

i = 0

def main(kernel_size, standard_deviation, delta_tresh, min_area):
    global i
    i += 1
    video = cv2.VideoCapture("F:\Disk D\PythonProjects\\visionLabs\data\srcvid.mp4", cv2.CAP_ANY)
    ret, frame = video.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), standard_deviation)
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('result' + str(i) + '.mp4', fourcc, 25, (w, h))
    print('Обрабатываю видео...')
    while True:
        # Сохраняем предыдущий кадр
        old_img = img.copy()

        # Проверка кадра
        check, frame = video.read()
        if not check:
            break

        # Переводим кадр в серый и сглаживаем
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), standard_deviation)

        # Модуль разницы двух кадров
        frame_diff = cv2.absdiff(img, old_img)

        # Фильтр если значение элемента меньше трешхолда то ставим 0, иначе 255
        # cv2.THRESH_BINARY - задает функцию по которой фильтруются элементы
        # Возвращает значение трешхолда и отфильтрованый кадр
        thresh = cv2.threshold(frame_diff, delta_tresh, 255, cv2.THRESH_BINARY)[1]

        # Выделение котуров
        # cv2.RETR_EXTERNAL - выделяет невложенные контуры
        # cv2.CHAIN_APPROX_SIMPLE - сжимает горизонтальные вертикальные и диагональные сегменты и возвращает меньше точек контуров
        (contors, hierarchy) = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contr in contors:
            # Площадь контура
            area = cv2.contourArea(contr)
            # Если меньше то скипаем
            if area < min_area:
                continue
            # Иначе фиксируем движение
            video_writer.write(frame)
    # Конец записи
    video_writer.release()
    print('Обработка завершена')

kernel_size = 3
standard_deviation = 50
delta_tresh = 40
min_area = 5
main(kernel_size, standard_deviation, delta_tresh, min_area)
