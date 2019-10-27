import cv2
from image_pyramid import Image_Pyramid
from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import train_test_split
import os
from fnmatch import fnmatch
from joblib import dump, load


class Feature_Extractor:
    def __init__(self, function, **args):
        self.__function__ = function
        self.__args__ = args

    def __yield_from_dir__(self, directory, ext):
        '''
        Returns a generator of images from source_dir resized to dsize.

        Parameters:
            source_dir (str): Source path for images.
            ext (str): Extension of images in directory (Default = ppm)
        '''
        for filename in os.listdir(pos_dir):
            if fnmatch(filename, ext):
                yield cv2.imread(os.path.join(pos_dir, filename), cv2.IMREAD_COLOR)

    def __apply_to_gen__(self, im_generator):
        '''
        Applies a function to all images fetched from a generator.
        Image will be passed as the first argument to the function.
        Used mainly as function for extracting features.
        Returns a list of results from each function run.

        Parameters:
            im_generator (generator): Generator of images, usually a return value from 'yield_images_from_dir'.
            function (function): Function to apply on all images.
            args (dict): Dictionary of arguments passed to the function, along with an image.
        '''
        results = []
        for image in im_generator:
            results.append(self.__function__(image, self.__args__))
        return results

    def dir_extract(self, dirpath, ext='*.ppm'):
        generator = self.__yield_from_dir__(dirpath, ext)
        results = []
        for image in generator:
            x, y, z = image.shape
            pyramid = Image_Pyramid(image, 1.0, ux, y))
            window_gen=pyramid.sliding_window((32, 32), (10, 10))
            results.append(self.__apply_to_gen__(window_gen))
        return results

    def extract(self, image):
        return self.__function__(image, self.__args__)


class SVM:
    def __init__(self, kernel = 'sigmoid', model_path = ''):
        if model_path:
            self.__clf__=load(model_path)
        else:
            self.__clf__=svm.SVC(kernel = 'sigmoid', verbose = True)

    def fit_n_test(self, x, y, split = 0.2, **args):
        train_features, test_features, train_labels, test_labels=train_test_split(
            features,
            classes,
            test_size = 0.2,
            random_state = 41)
        self.__clf__.fit(train_features, train_labels, args)
        return self.__test__(test_features, test_labels)

    def __test__(self, test_x, test_y):
        for index, tf in enumerate(test_x):
            if clf.predict([tf]) == test_y[index]:
                counter += 1
        return (counter / len(test_y)) * 100

    def save(self, filepath):
        dump(self.__clf__, filepath)

    def predict(image, feature_gen):
        features=feature_gen.extract(image)
        return self.__clf__.predict([featuress])


def main():
    feature_extr=Feature_Extractor(hog, orientations = 9, pixels_per_cell = (
        4, 4), cells_per_block = (2, 2), multichannel = True)

    features=feature_extr.dir_extract('./data/detection/train/positive')
    classes=[1] * len(features)
    neg_features=feature_extr.dir_extract('./data/detection/train/negative')
    classes.extend([0] * len(neg_features))
    features.extend(neg_features)

    clf=SVM()
    accuracy=clf.fit_n_test(features, classes)
    print(f"Train-test accuracy: {accuracy}%")


if __name__ == "__main__":
    main()
