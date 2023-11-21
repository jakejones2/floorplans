import numpy as np
import cv2

img = cv2.imread("floorplan1.png", cv2.IMREAD_GRAYSCALE)

thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))  # 5 5
mask = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

# mask = cv2.bitwise_not(mask)

cv2.imwrite("floorplan1_walls.jpg", mask)

# https://stackoverflow.com/questions/75459101/how-to-apply-erosion-for-only-thick-lines-in-images
