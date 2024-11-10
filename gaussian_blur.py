import cv2

def gaussian_blur():
    file = r'E:\python\visionLabs\data\Adler2009.jpg'
    image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    if image is None:
        print("Invalid file: " + file)
        exit(-1)
    gauss3x3 = cv2.GaussianBlur(image, (3, 3), 0)
    gauss5x5 = cv2.GaussianBlur(image, (5, 5), 0)
    window_name = "No Gauss"
    window_name_filtered3x3 = "Gauss 3x3"
    window_name_filtered5x5 = "Gauss 5x5"
    cv2.namedWindow(window_name)
    cv2.namedWindow(window_name_filtered3x3)
    cv2.namedWindow(window_name_filtered5x5)
    cv2.imshow(window_name, image)
    cv2.imshow(window_name_filtered3x3, gauss3x3)
    cv2.imshow(window_name_filtered5x5, gauss5x5)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
