import cv2
from image_pyramid import Image_Pyramid
from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import train_test_split
import os
from fnmatch import fnmatch
from joblib import dump, load


class Feature_Extractor:
    def __init__(self, function, *args, **kwargs):
        self.__function__ = function
        self.__args__ = args
        self.__kwargs__ = kwargs

    def __yield_from_dir__(self, directory, ext):
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
                image = cv2.resize(image, (32, 32))
                yield image

    def dir_extract(self, dirpath, ext='*.ppm'):
        '''
        Extracts features from all images within a directory,\
            using the function specified in Feature_Extractor __init__.
        Returns a list of results of a 
        '''
        generator = self.__yield_from_dir__(dirpath, ext)
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
            self.__clf__ = svm.SVC(kernel='sigmoid', verbose=True)

    def fit_n_test(self, x, y, split=0.2, *args, **kwargs):
        train_features, test_features, train_labels, test_labels = train_test_split(
            x,
            y,
            test_size=split,
            random_state=41)
        self.__clf__.fit(train_features, train_labels, *args, **kwargs)
        return self.__test__(test_features, test_labels)

    def __test__(self, test_x, test_y):
        counter = 0
        for index, tf in enumerate(test_x):
            if self.__clf__.predict([tf]) == test_y[index]:
                counter += 1
        return (counter / len(test_y)) * 100

    def save(self, filepath):
        dump(self.__clf__, filepath)

    def predict(image, feature_gen):
        features = feature_gen.extract(image)
        return self.__clf__.predict([featuress])


def main():
    feature_extr = Feature_Extractor(hog, orientations=9, pixels_per_cell=(
        4, 4), cells_per_block=(2, 2), multichannel=True)

    print("Extracting features from the trainig set.")
    features = feature_extr.dir_extract('./data/detection/train/positive')
    classes = [1] * len(features)
    neg_features = feature_extr.dir_extract('./data/detection/train/negative')
    classes.extend([0] * len(neg_features))
    features.extend(neg_features)

    clf = SVM()
    print("Fitting the SVM classifier.")
    accuracy = clf.fit_n_test(features, classes)
    print(f"Train-test accuracy: {accuracy}%")


if __name__ == "__main__":
    main()
