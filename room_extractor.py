import numpy as np
import cv2

PATH = "images/ex1/"

img = cv2.imread(f"{PATH}rooms.jpg", cv2.IMREAD_GRAYSCALE)
horizontal_crop = img.copy()
vertical_crop = img.copy()

# delete pixels from left across
for row in horizontal_crop:
    pix = 0
    while True:
        if pix == len(row):
            break
        if row[pix]:
            row[pix] = 0
        else:
            break
        pix += 1

# delete pixels from right across
for row in horizontal_crop:
    pix = len(row) - 1
    while True:
        if pix == len(row):
            break
        if row[pix]:
            row[pix] = 0
        else:
            break
        pix -= 1

# delete pixels from top down
for i in range(len(vertical_crop[0])):
    pix = 0
    while True:
        if pix == len(vertical_crop):
            break
        extra = 1 if i < len(vertical_crop[0]) - 1 else 0
        if vertical_crop[pix][i] and vertical_crop[pix][i + extra]:
            vertical_crop[pix][i] = 0
        else:
            break
        pix += 1

# delete pixels from bottom up
for i in range(len(vertical_crop[0])):
    pix = len(vertical_crop) - 1
    while True:
        extra = 1 if i < len(vertical_crop[0]) - 1 else 0
        if vertical_crop[pix][i] and vertical_crop[pix][i + extra]:
            vertical_crop[pix][i] = 0
        else:
            break
        pix -= 1

cropped = img.copy()

# combine crops - removes white space on edge of image
for row in range(len(img)):
    for pix in range(len(img[0])):
        if not vertical_crop[row][pix] or not horizontal_crop[row][pix]:
            cropped[row][pix] = 0

cropped = cv2.threshold(cropped, 120, 255, cv2.THRESH_BINARY)[1]

cv2.imwrite(f"{PATH}cropped.jpg", cropped)

img = cropped.copy()
original = cv2.imread(f"{PATH}original.png", cv2.IMREAD_GRAYSCALE)

# before continuing, try to remove the walls from the original? so we just see objects

cnts = cv2.findContours(cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

image_number = 0
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    if w < 10 or h < 10:
        continue
    cv2.rectangle(cropped, (x, y), (x + w, y + h), (0, 255, 0), 3)
    ROI = original[y : y + h, x : x + w]
    cv2.imwrite(f"{PATH}rooms/ROI_{image_number}.png", ROI)
    image_number += 1

cv2.imwrite(f"{PATH}final.jpg", cropped)

# for row in range(len(img)):
#     for pix in range(len(img[0])):
#         if pix:
#
