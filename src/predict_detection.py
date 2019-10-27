from joblib import load
from skimage.feature import hog
from sklearn import svm
import detection as dt
import cv2


clf = load('models/detection.joblib')
image = cv2.imread('data/detection/dirty/00268.ppm', cv2.IMREAD_COLOR)
x, y, z = image.shape
pyramid = dt.Image_Pyramid(image, 1.0, (x, y))
generator = pyramid.sliding_window((32, 32), (10, 10))

outputs = []
print("Initialized SVM")
for item in generator:
    fd = hog(item[0], orientations=9, pixels_per_cell=(
        4, 4), cells_per_block=(2, 2), multichannel=True)
    result = clf.predict([fd])
    if result == 1:
        x1, y1, x2, y2 = item[1]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

cv2.imwrite('output.png', image)