import cv2
import numpy as np


def log_matrix(matrix, size: int, acc: int = 2):
    for i in range(size):
        for j in range(size):
            print(f"{matrix[i, j]:.{acc}f}", end='\t')
        print()


def gauss_function(x: int, y: int, sigma: float, mx: float, my: float) -> float:
    return 1 / (2 * np.pi * sigma ** 2) * np.e ** (-((x - mx) ** 2 + (y - my) ** 2) / (2 * sigma ** 2))


def gauss_kernel(size: int, sigma: float, mx: float, my: float | None = None):
    if size < 0 or size % 2 == 0:
        raise ValueError("ERROR: Size of a kernel must be odd and above 0")
    if my is None:
        my = mx
    kernel = np.empty((size, size))
    for i in range(size):
        for j in range(size):
            kernel[i, j] = gauss_function(i, j, sigma, mx, my)
    log_matrix(kernel, size)
    return kernel / np.sum(kernel)


def convolution(image, kernel, size: int):
    h, w = image.shape[:2]
    margin = size // 2
    blurred = np.zeros((h, w, 3), np.uint8)
    for y in range(margin, h - margin):
        for x in range(margin, w - margin):
            b = g = r = 0
            for y0 in range(-margin, margin + 1):
                for x0 in range(-margin, margin + 1):
                    img_y = y + y0
                    img_x = x + x0
                    kernel_val = kernel[y0 + margin][x0 + margin]
                    img_bgr = image[img_y, img_x]
                    b += img_bgr[0] * kernel_val
                    g += img_bgr[1] * kernel_val
                    r += img_bgr[2] * kernel_val
            blurred[y, x] = (b, g, r)
    return blurred[margin:h - margin, margin:w - margin]


def gauss_manual(size: int, sigma: float):
    file = r'E:\python\visionLabs\data\Adler2009.jpg'
    image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    if image is None:
        print("Invalid file: " + file)
        exit(-1)
    gauss_name = f"Gauss blur {size} x {size}"
    print(gauss_name)
    kernel = gauss_kernel(size, sigma, size // 2)
    blurred = convolution(image, kernel, size)
    cv2.namedWindow("No effect")
    cv2.imshow("No effect", image)
    cv2.namedWindow(gauss_name)
    cv2.imshow(gauss_name, blurred)
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    gauss_manual(3, 1.5)
    gauss_manual(5, 1.5)

