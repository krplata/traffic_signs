
import os  # noqa
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # noqa
import tensorflow as tf  # noqa

import numpy as np
import pandas as pd
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import compare_pred as cp


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

    def run_training(self, train_gen, valid_gen, eps=10):
        self.__model__.fit_generator(train_gen, epochs=eps,
                                     validation_data=valid_gen, verbose=1)
        self.__model__.evaluate_generator(generator=valid_gen)

    def predict_w_gen(self, test_gen, step_size):
        test_gen.reset()
        return self.__model__.predict_generator(
            test_gen, steps=step_size, verbose=1)

    def predictions_to_csv(self, test_gen, predictions, filename, delimiter=';'):
        predicted_class_indices = np.argmax(predictions, axis=1)
        labels = (train_flow.class_indices)
        labels = dict((v, k) for k, v in labels.items())
        preds = [labels[k] for k in predicted_class_indices]
        filenames = test_gen.filenames
        results = pd.DataFrame({"Filename": filenames,
                                "Predictions": preds})
        results.to_csv("results.csv", index=False, sep=delimiter)


train_flow = ImageDataGenerator().flow_from_directory(
    './data/train/', class_mode='categorical', batch_size=64, color_mode="rgb", shuffle=True, target_size=(31, 31))
val_flow = ImageDataGenerator().flow_from_directory(
    './data/validate/', class_mode='categorical', batch_size=64, color_mode="rgb", shuffle=True, target_size=(31, 31))
test_flow = ImageDataGenerator().flow_from_directory(
    './data/test/', class_mode='categorical', batch_size=1, color_mode="rgb", shuffle=False, target_size=(31, 31))

tsmodel = TsRecognitionModel()
tsmodel.run_training(train_flow, val_flow)

print("Running predictions on internal test data:")
STEP_SIZE_TEST = test_flow.n//test_flow.batch_size
predictions = tsmodel.predict_w_gen(test_flow, STEP_SIZE_TEST)
tsmodel.predictions_to_csv(test_flow, predictions, 'results.csv')
cp.accuracy_on_generated("results.csv")

print("Results of predictions on external test data:")
