import cv2
import glob

dataset_dir = "data/challenge_augmented"

image_file_paths = []

for file_path in glob.glob(dataset_dir + "/*.png"):
    image_file_paths.append(file_path)

for image_file_path in image_file_paths:

    image = cv2.imread(image_file_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (128, 128, 128), 1)
    cv2.imwrite(image_file_path, image)
