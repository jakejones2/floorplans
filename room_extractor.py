"""
This script extracts rooms a floorplan based on a "rooms" image of its enclosed 
rooms (see wall_joiner.py). Provide a PATH to store images and prevent files 
being overwritten.

The image is first cropped manually to remove the outer borders. The implementation
of this is clumsy and homemade - need to find a proper algorithm at some point! 

See https://stackoverflow.com/questions/57777368/how-to-detect-and-extract-rectangles-with-python-opencv
"""
import cv2
from pathlib import Path

PATH = "images/ex3/"

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

cv2.imwrite(f"{PATH}6_cropped.jpg", cropped)

# load original for chopping up into rooms
original = cv2.imread(f"{PATH}original.jpg", cv2.IMREAD_GRAYSCALE)
img = cropped.copy()

# find contours
cnts = cv2.findContours(cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

# create regions of interest (rooms)
Path(f"{PATH}rooms").mkdir(parents=True, exist_ok=True)
image_number = 0
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    if w < 10 or h < 10:
        continue
    cv2.rectangle(cropped, (x, y), (x + w, y + h), (0, 255, 0), 3)
    ROI = original[y : y + h, x : x + w]
    cv2.imwrite(f"{PATH}rooms/ROI_{image_number}.png", ROI)
    image_number += 1

cv2.imwrite(f"{PATH}7_final.jpg", cropped)
