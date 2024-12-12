import cv2
import sys


def main():
    # имя картинки задаётся первым параметром
    filename = sys.argv[1] if len(sys.argv) >= 2 else "data/PIWO.jpg"

    # получаем картинку в градациях серого
    default = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    src = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    print(f"[i] image: {filename}")
    assert src is not None, "Image not found!"

    # показываем изображение
    cv2.namedWindow("original", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("original", src)

    # получаем бинарное изображение с помощью детектора Кэнни
    dst2 = cv2.Canny(src, 50, 200)

    cv2.namedWindow("bin", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("bin", dst2)

    # вычитаем контуры из оригинального изображения
    subtracted = cv2.subtract(src, dst2)

    cv2.namedWindow("sub", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("sub", subtracted)

    # ждём нажатия клавиши
    cv2.waitKey(0)

    # освобождаем ресурсы и удаляем окна
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
