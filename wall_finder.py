"""
This script takes a floor plan and extracts the walls, producing a new image.
Provide a PATH to store images and an initial image titled "1_original.jpg".
This is a very basic implementation, the kernals and thresholds will likely need adjusting.

See https://stackoverflow.com/questions/75459101/how-to-apply-erosion-for-only-thick-lines-in-images
"""

import cv2

PATH = "images/ex3/"

img = cv2.imread(f"{PATH}1_original.jpg", cv2.IMREAD_GRAYSCALE)

thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# TO DO: kernel size needs adjusting depending on wall thickness
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
mask = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

# mask = cv2.bitwise_not(mask)

cv2.imwrite(f"{PATH}2_walls.jpg", mask)
