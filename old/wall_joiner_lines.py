import numpy as np
import cv2
import math
import random

img = cv2.imread("floorplan1_walls.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# inverse binary image
th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

cv2.imwrite("th.jpg", th)

# skeletonize
sk = cv2.ximgproc.thinning(th, None, 1)

cv2.imwrite("sk.jpg", sk)

dst = cv2.Canny(sk, 50, 200, None, 3)

# Copy edges to the images that will display the results in BGR
cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
cdstP = np.copy(cdst)

cv2.imwrite("dst.jpg", dst)

lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

# Draw the lines
if lines is not None:
    # NEXT - ELIMINATE LOOSE ENDS AFTER LINE DRAWN - ONLY CAN BE 'USED' ONCE

    brightness = 255
    for i in range(0, len(lines)):
        brightness = brightness - 20
        if brightness < 0:
            brightness = 0
        rand1 = random.randint(0, 255)
        rand2 = random.randint(0, 255)
        rand3 = random.randint(0, 255)

        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv2.line(cdst, pt1, pt2, (255 - brightness, 0, brightness), 3, cv2.LINE_AA)
        if i > 3:
            # break
            continue

linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

cv2.imwrite("v1.jpg", cdst)
cv2.imwrite("v2.jpg", cdstP)
