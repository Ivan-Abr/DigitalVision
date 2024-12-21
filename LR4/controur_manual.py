import cv2
import numpy as np

def convolution(img, kernel):
    kernel_size = len(kernel)

    x_start = kernel_size // 2
    y_start = kernel_size // 2

    matrix = np.zeros(img.shape)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            matrix[i][j] = img[i][j]

    for i in range(x_start, len(matrix)-x_start):
        for j in range(y_start, len(matrix[i])-y_start):

            value = 0
            for k in range(-(kernel_size//2), kernel_size//2+1):
                for l in range(-(kernel_size//2), kernel_size//2+1):
                    value += int(img[i + k][j + l]) * kernel[k + (kernel_size//2)][l + (kernel_size//2)]
            matrix[i][j] = value

    return matrix

def log_matrix(matrix, size: int, acc: int = 2):
    for i in range(size):
        for j in range(size):
            print(f"{matrix[i, j]:.{acc}f}", end='\t')
        print()

def angle_num(x, y):
    tg = y/x if x != 0 else 999

    if (x < 0):
        if (y < 0):
            if (tg > 2.414):
                return 0
            elif (tg < 0.414):
                return 6
            elif (tg <= 2.414):
                return 7
        else:
            if (tg < -2.414):
                return 4
            elif (tg < -0.414):
                return 5
            elif (tg >= -0.414):
                return 6
    else:
        if (y < 0):
            if (tg < -2.414):
                return 0
            elif (tg < -0.414):
                return 1
            elif (tg >= -0.414):
                return 2
        else:
            if (tg < 0.414):
                return 2
            elif (tg < 2.414):
                return 3
            elif (tg >= 2.414):
                return 4

def main(path, standard_deviation, ksize, bound_path):

    # Предварительная подготовка изображения
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    gauss_blurred = cv2.GaussianBlur(img, (ksize, ksize), standard_deviation)
    cv2.namedWindow("Image_gauss")
    cv2.imshow("Image_gauss", gauss_blurred)
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()

    # Операторы Собеля
    gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    img_gx = convolution(img, gx)
    img_gy = convolution(img, gy)

    le_gradient = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            le_gradient[i][j] = img[i][j]

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            le_gradient[i][j] = np.sqrt(img_gx[i][j] ** 2 + img_gy[i][j] * 2)

    angles = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            angles[i][j] = angle_num(img_gx[i][j], img_gy[i][j])

    img_gradient = img.copy()
    max_gradient = np.max(le_gradient)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # нормализация значения градиента относительно макс значения и масшабирования в диапозоне от 0 до 255
            img_gradient[i][j] = (float(le_gradient[i][j]) / max_gradient) * 255
    log_matrix(img_gradient, img_gradient.size)
    cv2.imshow('img gradient to print ', img_gradient)
if __name__ == "__main__":
    main('../data/PIWO.jpg', 6, 3, 6)
