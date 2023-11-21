import numpy as np
import cv2
import random

PATH = "images/ex1/"

img = cv2.imread(f"{PATH}walls.jpg")
ends = img.copy()
boxes = img.copy()
intersections = img.copy()
fillings = img.copy()

base = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
for pt in pts:
    ends = cv2.circle(ends, (pt[1], pt[0]), 3, (0, 200, 200), -1)

cv2.imwrite(f"{PATH}loose_ends.jpg", ends)

# convert array of points to list of tuples
pts = list(map(tuple, pts))

# final is an array of rows of pixels, length being img height
final = sk.copy()

# create box around each loose end, find intersection with wall
for i, pt1 in enumerate(pts):
    search_dist = 10
    search_box = []
    intersection = None
    while True:
        for dist in range(2 * search_dist + 1):
            search_box.append(
                [pt1[0] - search_dist + dist, pt1[1] - search_dist]
            )  # top line
            search_box.append(
                [pt1[0] - search_dist + dist, pt1[1] + search_dist]
            )  # bottom line
            search_box.append(
                [pt1[0] - search_dist, pt1[1] - search_dist + dist]
            )  # left side
            search_box.append(
                [pt1[0] + search_dist, pt1[1] - search_dist + dist]
            )  # right side
        for point in search_box:
            boxes[point[0]][point[1]] = (0, 0, 255)
            if final[point[0]][point[1]]:
                intersection = point
                intersections[point[0]][point[1]] = (0, 255, 0)
        if intersection:
            break
        else:
            search_dist += 1

    # draw intersection
    intersections = cv2.circle(
        boxes, (intersection[1], intersection[0]), 3, (0, 200, 200), -1
    )

    dy = pt1[0] - intersection[0]
    dx = pt1[1] - intersection[1]

    # use these in fillings to debug
    rand1 = random.randint(0, 255)
    rand2 = random.randint(0, 255)
    rand3 = random.randint(0, 255)

    # TO DO: somehow x and y are getting swapped throughout... Makes logic confusing!
    print("dy", dy, dx)
    print("pt1", pt1)
    print("int", intersection)

    # extend walls horizontally or vertically until they reach another wall
    if abs(dy) < abs(dx):  # TO DO: if dy == 0
        extension = 1
        direction = -1 if dy > 0 else 1
        while True:
            new_x = pt1[1] + (extension * direction)
            if abs(new_x) >= len(final[0]):
                break
            if final[pt1[0]][new_x] > 30:
                fillings = cv2.line(
                    fillings,
                    (pt1[1], pt1[0]),
                    (new_x, pt1[0]),
                    (0, 0, 0),
                    thickness=3,
                )
                break

            extension += 1
    else:  # TO DO: if dx == 0
        extension = 1
        direction = 1 if dy > 0 else -1
        while True:
            new_y = pt1[0] + (extension * direction)
            if abs(new_y) >= len(final):
                break
            if final[new_y][pt1[1]] > 30:
                fillings = cv2.line(
                    fillings,
                    (pt1[1], pt1[0]),
                    (pt1[1], new_y),
                    (0, 0, 0),
                    thickness=3,
                )
                break
            extension += 1

cv2.imwrite(f"{PATH}boxes.jpg", boxes)
cv2.imwrite(f"{PATH}rooms.jpg", fillings)
