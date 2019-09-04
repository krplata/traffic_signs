import numpy as np
import os
import cv2


def noiseImage(image):
    mean, sigma = 0.1, 0.5
    rand = np.random.normal(mean, sigma, image.shape)
    return rand


def readImage(filepath):
    return cv2.imread(filepath)


image = readImage("./ex2.jpg")
noised = noiseImage(image).astype(np.uint8)
# both = np.concatenate(image, noised)
print(noised)
# cv2.GaussianBlur(tuple(image), 5, 0.25, tuple(blured))
cv2.imshow("noised", noised)
# cv2.namedWindow("noised", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("noised", 800, 600)
cv2.waitKey(0)
