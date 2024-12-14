import cv2
import numpy as np

def find_gradient(image):
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    gx = cv2.filter2D(image, cv2.CV_32F, sobel_x)
    gy = cv2.filter2D(image, cv2.CV_32F, sobel_y)

    gradient_length = np.sqrt(gx ** 2 + gy ** 2)
    gradient_direction = np.arctan2(gy, gx) * (180 / np.pi)
    return gradient_length, gradient_direction


def suppress_nonmax(image, gradient_length, gradient_direction):
    height, width = image.shape
    suppressed = np.zeros((height, width), dtype=np.float32)

    angle = gradient_direction % 180

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            try:
                q = 255
                r = 255

                if (0 <= angle[y, x] < 22.5) or (157.5 <= angle[y, x] <= 180):
                    q = gradient_length[y, x + 1]
                    r = gradient_length[y, x - 1]
                elif 22.5 <= angle[y, x] < 67.5:
                    q = gradient_length[y + 1, x - 1]
                    r = gradient_length[y - 1, x + 1]
                elif 67.5 <= angle[y, x] < 112.5:
                    q = gradient_length[y + 1, x]
                    r = gradient_length[y - 1, x]
                elif 112.5 <= angle[y, x] < 157.5:
                    q = gradient_length[y - 1, x - 1]
                    r = gradient_length[y + 1, x + 1]

                if (gradient_length[y, x] >= q) and (gradient_length[y, x] >= r):
                    suppressed[y, x] = gradient_length[y, x]
                else:
                    suppressed[y, x] = 0
            except IndexError:
                pass

    return suppressed


def double_threshold(image, low_threshold, high_threshold):
    strong = 255
    weak = 50

    strong_pixels = (image >= high_threshold)
    weak_pixels = ((image >= low_threshold) & (image < high_threshold))

    thresholded = np.zeros_like(image, dtype=np.uint8)
    thresholded[strong_pixels] = strong
    thresholded[weak_pixels] = weak

    return thresholded





def main():
    path = 'data/PIWO.jpg'
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    ksize = 3
    standard_deviation = 6
    blurred_img = cv2.GaussianBlur(img, (ksize, ksize), standard_deviation)
    cv2.namedWindow("Image_gauss")
    cv2.imshow("Image_gauss", blurred_img)

    gradient_length, gradient_direction = find_gradient(blurred_img)
    cv2.namedWindow("Gradient length")
    cv2.imshow("Gradient_length", gradient_length)
    cv2.namedWindow("Gradient_direction")
    cv2.imshow("Gradient_direction", gradient_direction)
    suppressed_image = suppress_nonmax(blurred_img, gradient_length, gradient_direction)
    suppressed_image = np.uint8(suppressed_image)
    cv2.imshow('Suppressed Image', suppressed_image)

    # тут должна быть максимальная длина градиента
    max_length = gradient_length.max()
    low_threshold, high_threshold = max_length * 0.05, max_length * 0.1
    filtered_image = double_threshold(suppressed_image, low_threshold,high_threshold)
    cv2.imshow('Final Image', filtered_image)
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
