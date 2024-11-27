import cv2
import numpy as np


def main(path, standard_deviation, ksize, bound_path):
    path = 'data/PIWO.jpg'
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    gauss_blurred = cv2.GaussianBlur(img, (ksize, ksize), standard_deviation)
    cv2.namedWindow("Image_gauss")
    cv2.imshow("Image_gauss", gauss_blurred)
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main('data/PIWO.jpg', 6, 3, 6)
