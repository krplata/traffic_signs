import os
import cv2
from fnmatch import fnmatch
from joblib import dump, load
from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold, cross_val_score, cross_val_predict
from image_pyramid import Image_Pyramid
from sklearn.metrics import confusion_matrix
import numpy as np


class Feature_Extractor:
    def __init__(self, function, *args, **kwargs):
        self.__function__ = function
        self.__args__ = args
        self.__kwargs__ = kwargs

    def __yield_from_dir__(self, directory, ext, grayscale=False, dsize=None):
        '''
        Returns a generator of images from source_dir resized to dsize.

        Parameters:
            source_dir (str): Source path for images.
            ext (str): Extension of images in directory (Default = ppm)
        '''
        for filename in os.listdir(directory):
            if fnmatch(filename, ext) or fnmatch(filename, '*.jpg'):
                image = cv2.imread(os.path.join(
                    directory, filename), cv2.IMREAD_COLOR)
                if dsize:
                    image = cv2.resize(image, dsize)
                if grayscale:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                yield image

    def dir_extract(self, dirpath, ext='*.ppm', grayscale=False, dsize=None):
        '''
        Extracts features from all images within a directory,\
            using the function specified in Feature_Extractor __init__.
        Returns a list of results of a
        '''
        generator = self.__yield_from_dir__(dirpath, ext, grayscale, dsize)
        results = []
        for image in generator:
            results.append(self.extract(image))
        return results

    def extract(self, image):
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.__function__(image, *self.__args__, **self.__kwargs__)


class SVM:
    def __init__(self, kernel='rbf', model_path=''):
        if model_path:
            print('xddd')
            self.__clf__ = load(model_path)
        else:
            self.__clf__ = svm.SVC(kernel=kernel, verbose=True)

    def fit(self, x, y, *args, **kwargs):
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, random_state=0)
        y_pred = self.__clf__.fit(x_train, y_train).predict(x_test)
        conf_mat = confusion_matrix(y_test, y_pred)
        print(conf_mat)

    def grid_search(self, x, y):
        param_grid = {
            "kernel": ["rbf"],
            "C": [0.01, 0.1, 1, 10, 100, 1000, 10000],
            "gamma": [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        }
        rkf = RepeatedKFold(n_splits=2, n_repeats=3)
        gs = GridSearchCV(svm.SVC(), param_grid,
                          scoring='accuracy', n_jobs=-1, cv=rkf)
        gs.fit(x, y)
        return gs.cv_results_

    def save(self, filepath):
        dump(self.__clf__, filepath)

    def predict(self, image, feature_gen):
        features = feature_gen.extract(image)
        return self.__clf__.predict([features])


class Detector:
    def __init__(self, circles_path='models/detect_circles.joblib', triangles_path='models/detect_triangles.joblib'):
        self.__tr_clf__ = SVM(model_path=triangles_path)
        self.__circle_clf__ = SVM(model_path=circles_path)

    def predict(self, image_path, feature_gen):
        full_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)
        pyramid = Image_Pyramid(full_image, 0.7, (300, 150))
        generator = pyramid.sliding_window((32, 32), (16, 16))
        index = 0
        for image in generator:
            index += 1
            triangle_resp = self.__tr_clf__.predict(image, feature_gen)
            circle_resp = self.__circle_clf__.predict(image, feature_gen)
            if (circle_resp[0] == 1 or triangle_resp[0] == 1) and circle_resp[0] != triangle_resp[0]:
                cv2.imwrite(f"output/im{index}.jpg", image)


def main():
    feature_extr = Feature_Extractor(hog, orientations=9, pixels_per_cell=(
        8, 8), cells_per_block=(2, 2))

    det = Detector()
    det.predict('scene_ex.png', feature_extr)


if __name__ == "__main__":
    main()
