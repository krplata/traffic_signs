import cv2
from image_pyramid import Image_Pyramid
from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import train_test_split
import os
from fnmatch import fnmatch
from joblib import dump, load
import time
from sklearn.model_selection import cross_val_score


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
            if fnmatch(filename, ext):
                image = cv2.imread(os.path.join(
                    directory, filename), cv2.IMREAD_COLOR)
                if dsize:
                    image = cv2.resize(image, dsize)
                if grayscale:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.Canny(image, 50, 200)
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
        return self.__function__(image, *self.__args__, **self.__kwargs__)


class SVM:
    def __init__(self, kernel='sigmoid', model_path=''):
        if model_path:
            self.__clf__ = load(model_path)
        else:
            self.__clf__ = svm.SVC(kernel=kernel, verbose=True)

    def fit(self, x, y, split=0.2, *args, **kwargs):
        train_features, test_features, train_labels, test_labels = train_test_split(
            x,
            y,
            test_size=split,
            random_state=41)
        self.__clf__.fit(train_features, train_labels, *args, **kwargs)

    def est_cross_val(self, x, y, k=5):
        return cross_val_score(self.__clf__, x, y, cv=k, n_jobs=-1)

    def save(self, filepath):
        dump(self.__clf__, filepath)

    def predict(self, image, feature_gen):
        image = cv2.Canny(image, 50, 200)
        features = feature_gen.extract(image)
        return self.__clf__.predict([features])


def main():
    feature_extr = Feature_Extractor(hog, orientations=9, pixels_per_cell=(
        8, 8), cells_per_block=(2, 2))

    print("Extracting features from the trainig set.")
    features = feature_extr.dir_extract(
        './data/detection/train/positive', dsize=(32, 32))
    classes = [1] * len(features)
    neg_features = feature_extr.dir_extract(
        './data/detection/train/negative', dsize=(32, 32))
    classes.extend([0] * len(neg_features))
    features.extend(neg_features)

    clf = SVM()
    print("Fitting the SVM classifier.")
    accuracy = clf.est_cross_val(features, classes)
    print(accuracy)
    # clf = SVM(model_path='models/detection.joblib')
    # image = cv2.imread('data/detection/stock/00000.ppm', cv2.IMREAD_COLOR)
    # x, y, z = image.shape
    # pyramid = Image_Pyramid(image, 0.7, (x*0.6, y*0.6))
    # generator = pyramid.sliding_window((32, 32), (10, 10))
    # start = time.time()
    # count = 0
    # for image in generator:
    #     if clf.predict(image, feature_extr) == 1:
    #         count += 1
    #         print("Found")
    #         cv2.imwrite(f"predictions/{count}.jpg", image)
    # print(f"Count: {count}")
    # print(time.time() - start)


if __name__ == "__main__":
    main()
