
import os  # noqa
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # noqa
import tensorflow as tf  # noqa
from keras.utils import plot_model
import numpy as np
import pandas as pd
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import comparison as cp
import matplotlib.pyplot as plt
from keras import models
from keras.preprocessing import image
import matplotlib.image as mpimg

class TsRecognitionModel:
    def __init__(self, layers=[]):
        self.__layers__ = layers
        self.__model__ = Sequential()
        self.__build_model__()

    def __build_model__(self):
        if self.__layers__:
            for layer in self.__layers__:
                self.__model__.add(layer)
        else:
            self.__model__.add(Conv2D(32, (3, 3), padding='same',
                                      input_shape=(31, 31, 3), activation='relu'))
            self.__model__.add(Conv2D(32, (3, 3), activation='relu'))
            self.__model__.add(MaxPooling2D(pool_size=(2, 2)))
            self.__model__.add(Dropout(0.2))
            self.__model__.add(
                Conv2D(64, (3, 3), padding='same', activation='relu'))
            self.__model__.add(Conv2D(64, (3, 3), activation='relu'))
            self.__model__.add(MaxPooling2D(pool_size=(2, 2)))
            self.__model__.add(Dropout(0.2))
            self.__model__.add(Flatten())
            self.__model__.add(Dense(512, activation='relu'))
            self.__model__.add(Dropout(0.2))
            self.__model__.add(Dense(43, activation='softmax'))
        self.__model__.compile(
            loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        plot_model(self.__model__, to_file='model.pdf')

    def run_training(self, train_gen, valid_gen, eps=25):
        self.__model__.fit_generator(train_gen, epochs=eps,
                                     validation_data=valid_gen, verbose=1)
        self.__model__.evaluate_generator(generator=valid_gen)

    def predict_w_gen(self, test_gen, step_size):
        test_gen.reset()
        return self.__model__.predict_generator(
            test_gen, verbose=1)

    def predictions_to_csv(self, test_gen, predictions, filename, delimiter=';'):
        predicted_class_indices = np.argmax(predictions, axis=1)
        labels = (train_flow.class_indices)
        labels = dict((v, k) for k, v in labels.items())
        preds = [labels[k] for k in predicted_class_indices]
        filenames = test_gen.filenames
        results = pd.DataFrame({"Filename": filenames,
                                "Predictions": preds})
        results.to_csv(filename, index=False, sep=delimiter)

    def save_model(self, filepath):
        self.__model__.save(filepath)


print("Initializing training data:")
train_flow = ImageDataGenerator().flow_from_directory(
    './data/recognition/train/', class_mode='categorical', batch_size=64, color_mode="rgb", shuffle=True, target_size=(31, 31))
print("Initializing validation data (subset of training data with an 80:20 split):")
val_flow = ImageDataGenerator().flow_from_directory(
    './data/recognition/validate/', class_mode='categorical', batch_size=64, color_mode="rgb", shuffle=True, target_size=(31, 31))
print("Initializing internal testing data (subset of training data with an 80:20 split):")
test_flow = ImageDataGenerator().flow_from_directory(
    './data/recognition/test/', class_mode='categorical', batch_size=1, color_mode="rgb", shuffle=False, target_size=(31, 31))

tsmodel = TsRecognitionModel()
tsmodel.run_training(train_flow, val_flow, eps=21)
print("Running predictions on internal test data:")
STEP_SIZE_TEST = test_flow.n//test_flow.batch_size
predictions = tsmodel.predict_w_gen(test_flow, STEP_SIZE_TEST)
tsmodel.predictions_to_csv(test_flow, predictions, 'results.csv')
cp.accuracy_on_generated("results.csv")

print("Results of predictions on external test data:")
externaldf = pd.read_csv(
    "./data/recognition/external/GT-final_test.csv", dtype=str, sep=';')

extern_gen = ImageDataGenerator().flow_from_dataframe(
    dataframe=externaldf,
    directory="./data/recognition/external/",
    x_col="Filename",
    y_col=None,
    batch_size=1,
    seed=42,
    shuffle=False,
    class_mode=None,
    target_size=(31, 31))

external_predictions = tsmodel.predict_w_gen(extern_gen, STEP_SIZE_TEST)
tsmodel.predictions_to_csv(
    extern_gen, external_predictions, 'ext_results.csv')
cp.accuracy_on_external("./data/recognition/external/GT-final_test.csv", "ext_results.csv")

print("Saving full model architecture into models/working_example.h5")
tsmodel.save_model('./models/working_example.h5')
