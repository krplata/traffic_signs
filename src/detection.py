import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn import svm
import time
import os
from fnmatch import fnmatch
import numpy as np
from joblib import dump, load


class Image_Pyramid:
    '''
    Creates a tuple filled with images scaled down by a factor.
    Used in a sliding window method for detecting shapes (in this case traffic signs).

    Parameters:
        image (cv2_image): Image used for scaling down and creating the pyramid.
        scale_factor (float): Factor by which the dimensions will be resized at each iteration. (Range: (0, 1))
        min_dim (list(int, int)): Cutoff point for generating images. If the dimensions
        don't align with the scale factor, an image smaller than min_dim won't be created.
    '''

    def __init__(self, image, scale_factor, min_dim):
        self.__images__ = (image, )
        if scale_factor <= 0 or scale_factor >= 1:
            return
        while True:
            resized_width = int(self.__images__[-1].shape[1] * scale_factor)
            resized_height = int(self.__images__[-1].shape[0] * scale_factor)
            if resized_height > min_dim[1] and resized_width > min_dim[0]:
                resized = cv2.resize(
                    self.__images__[-1], dsize=(resized_width, resized_height))
                self.__images__ += (resized, )
            else:
                break

    def sliding_window(self, size, step):
        '''
        Runs a 'size' sized window with a 'step' over the image.
        On each iteration, the function yields a window available for further classification.

        Parameters:
            - size (x:int, y:int): Defines the dimensions of the sliding window.
            - step (x:int, y:int): Defines the step sizes along the horizontal and vertical axis.
        '''
        for index, image in enumerate(self.__images__):
            for y in range(0, image.shape[0] - size[1], step[1]):
                for x in range(0, image.shape[1] - size[0], step[0]):
                    yield [image[y:y+size[1], x:x+size[0]], (x, y, x+size[0], y+size[1])]


# pos_dir = './data/detection/train/positive'
# neg_dir = './data/detection/train/negative'

# features = []
# classes = []

# for filename in os.listdir(pos_dir):
#     if fnmatch(filename, '*.ppm'):
#         image = cv2.imread(os.path.join(pos_dir, filename), cv2.IMREAD_COLOR)
#         fd = hog(image, orientations=9, pixels_per_cell=(8, 8),
#                  cells_per_block=(2, 2), multichannel=True)
#         features.append(fd)

# pos_feature_count = len(features)
# classes = [1] * pos_feature_count

# for filename in os.listdir(neg_dir):
#     if fnmatch(filename, '*.jpg'):
#         image = cv2.imread(os.path.join(neg_dir, filename), cv2.IMREAD_COLOR)
#         fd = hog(image, orientations=9, pixels_per_cell=(8, 8),
#                  cells_per_block=(2, 2), multichannel=True)
#         features.append(fd)

# print("Fitting")
# classes.extend([0] * (len(features) - pos_feature_count))
# clf = svm.SVC(kernel='sigmoid', verbose=True)
# clf.fit(features, classes)
# im = cv2.imread(os.path.join(pos_dir, '3354800015_00020.ppm'))
# fd = hog(im, orientations=9, pixels_per_cell=(8, 8),
#          cells_per_block=(2, 2), multichannel=True)
# print("Prediction:")
# print(clf.predict([fd]))
# dump(clf, 'models/detection.joblib')
