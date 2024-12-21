import cv2

from graph_contour import GraphContour

images = [
    {
        "path": "F:\\Disk D\\PythonProjects\\visionLabs\\data\\dataset\\apple.jpg",
        "name": "apple",
    },
    {
        "path": "F:\\Disk D\\PythonProjects\\visionLabs\\data\\dataset\\apples.jpeg",
        "name": "apples",
    },
    {
        "path": "F:\\Disk D\\PythonProjects\\visionLabs\\data\\dataset\\many_apples.jpg",
        "name": "many_apples",
    }
]

deviations = [
    5, 1, 0.1
]

size_component_thresholds = [
    0, 100, 150
]

contrasts = [
    10, 15, 20
]

for deviation in deviations:
    for size_component_threshold in size_component_thresholds:
        for contrast in contrasts:
            graph_alg = GraphContour(
                image_size=(500, 500),
                image_show_list=[
                    # ImageShowGraphAlgorythmEnum.CONTRAST
                ],
                kernel_size=7,
                deviation=deviation,
                size_component_threshold=size_component_threshold,
                contrast=contrast,
            )

            for image in images:
                processed_image = graph_alg.process_image_with_return(image["path"])

                cv2.imwrite(
                    f"F:\\Disk D\\PythonProjects\\visionLabs\\data\\results\\{image["name"]}_gauss_{deviation}_size_component_threshold_{size_component_threshold}_contrast{contrast}.png",
                    processed_image
                )

                print(f"{image['name']} done")

cv2.waitKey(0)
cv2.destroyAllWindows()