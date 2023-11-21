import cv2

PATH = "images/ex3/"

img = cv2.imread(f"{PATH}original.jpg", cv2.IMREAD_GRAYSCALE)

thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# TO DO: kernel size needs adjusting depending on wall thickness
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
mask = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

# mask = cv2.bitwise_not(mask)

cv2.imwrite(f"{PATH}walls.jpg", mask)

# https://stackoverflow.com/questions/75459101/how-to-apply-erosion-for-only-thick-lines-in-images
