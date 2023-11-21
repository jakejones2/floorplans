import numpy as np
import cv2
import math
import random

img = cv2.imread("floorplan1_walls.jpg")
base = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

count = 0


# inverse binary image
th = cv2.threshold(base, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# skeletonize
sk = cv2.ximgproc.thinning(th, None, 1)

# Kernels for each of the 8 variations
k1 = np.array(([0, 0, 0], [-1, 1, -1], [-1, -1, -1]), dtype="int")
k2 = np.array(([0, -1, -1], [0, 1, -1], [0, -1, -1]), dtype="int")
k3 = np.array(([-1, -1, 0], [-1, 1, 0], [-1, -1, 0]), dtype="int")
k4 = np.array(([-1, -1, -1], [-1, 1, -1], [0, 0, 0]), dtype="int")

k5 = np.array(([-1, -1, -1], [-1, 1, -1], [0, -1, -1]), dtype="int")
k6 = np.array(([-1, -1, -1], [-1, 1, -1], [-1, -1, 0]), dtype="int")
k7 = np.array(([-1, -1, 0], [-1, 1, -1], [-1, -1, -1]), dtype="int")
k8 = np.array(([0, -1, -1], [-1, 1, -1], [-1, -1, -1]), dtype="int")

# hit-or-miss transform
o1 = cv2.morphologyEx(sk, cv2.MORPH_HITMISS, k1)
o2 = cv2.morphologyEx(sk, cv2.MORPH_HITMISS, k2)
o3 = cv2.morphologyEx(sk, cv2.MORPH_HITMISS, k3)
o4 = cv2.morphologyEx(sk, cv2.MORPH_HITMISS, k4)
out1 = o1 + o2 + o3 + o4

o5 = cv2.morphologyEx(sk, cv2.MORPH_HITMISS, k5)
o6 = cv2.morphologyEx(sk, cv2.MORPH_HITMISS, k6)
o7 = cv2.morphologyEx(sk, cv2.MORPH_HITMISS, k7)
o8 = cv2.morphologyEx(sk, cv2.MORPH_HITMISS, k8)
out2 = o5 + o6 + o7 + o8

# contains all the loose end points
out = cv2.add(out1, out2)

# store the loose end points and draw them for visualization
pts = np.argwhere(out == 255)
# loose_ends = img.copy()

count = count + 1

for pt in pts:
    base = cv2.circle(base, (pt[1], pt[0]), 2, (0, 0, 0), -1)


cv2.imwrite("loose_ends.jpg", base)

# convert array of points to list of tuples
pts = list(map(tuple, pts))

final = img.copy()

# iterate every point in the list and draw a line between nearest point in the same list
for i, pt1 in enumerate(pts):
    rand1 = random.randint(0, 255)
    rand2 = random.randint(0, 255)
    rand3 = random.randint(0, 255)

    min_dist = max(img.shape[:2])
    sub_pts = pts.copy()
    del sub_pts[i]
    pt_2 = sub_pts[0]
    for pt2 in sub_pts:
        dist = int(np.linalg.norm(np.array(pt1) - np.array(pt2)))
        # print(dist)
        if (abs(pt1[0] - pt2[0]) > 5) and (abs(pt1[1] - pt2[1]) > 5):
            continue
        if dist < min_dist:
            min_dist = dist
            pt_2 = pt2

    final = cv2.line(
        final, (pt1[1], pt1[0]), (pt_2[1], pt_2[0]), (rand1, rand2, rand3), thickness=2
    )

cv2.imwrite("floorplan1_rooms.jpg", final)
