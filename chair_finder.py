"""
This script attempts to find chairs within a room based on sample images.
Provide a PATH to the example you are working on, and create 'target' images to search for.

Currently not working! I think the images are too small and pixelated? Yet scaling doesn't help.
Check out images/ex1/chair-matching.jpg to see the problem.

See https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html
"""

import cv2
import numpy as np

PATH = "images/ex1/"
FLANN_INDEX_KDTREE = 1
MIN_MATCH_COUNT = 4

chair_small = cv2.imread(f"{PATH}targets/chair5.png", cv2.IMREAD_GRAYSCALE)
w, h = chair_small.shape
scale_factor = 3
chair_1 = cv2.resize(
    chair_small, (h * scale_factor, w * scale_factor), interpolation=cv2.INTER_AREA
)

room = cv2.imread(f"{PATH}rooms/ROI_3.png", cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(chair_1, None)
kp2, des2 = sift.detectAndCompute(room, None)

index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)


matches = flann.knnMatch(des1, des2, k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m, n in matches:
    good.append(m)
    # if m.distance < 1 * n.distance:  # this should be lower, e.g. 0.7
    #  good.append(m)

print("matches: ", len(good))

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h, w = chair_1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    img2 = cv2.polylines(room, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    draw_params = dict(
        matchColor=(0, 255, 0),  # draw matches in green color
        singlePointColor=None,
        matchesMask=matchesMask,  # draw only inliers
        flags=2,
    )

    img3 = cv2.drawMatches(chair_1, kp1, img2, kp2, good, None, **draw_params)

    cv2.imwrite(f"{PATH}chair-matching.jpg", img3)

else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None
