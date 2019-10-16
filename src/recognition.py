
import os  # noqa
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # noqa
import tensorflow as tf  # noqa

import numpy as np
import pandas as pd
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import compare_pred as cp

train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_flow = train_datagen.flow_from_directory(
    './data/train/', class_mode='categorical', batch_size=64, color_mode="rgb", shuffle=True, target_size=(31, 31))
val_flow = val_datagen.flow_from_directory(
    './data/validate/', class_mode='categorical', batch_size=64, color_mode="rgb", shuffle=True, target_size=(31, 31))
test_flow = test_datagen.flow_from_directory(
    './data/test/', class_mode='categorical', batch_size=1, color_mode="rgb", shuffle=False, target_size=(31, 31))

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(31, 31, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(43, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.fit_generator(train_flow, epochs=10, validation_data=val_flow, verbose=1)
model.evaluate_generator(generator=val_flow)

STEP_SIZE_TEST = test_flow.n//test_flow.batch_size
test_flow.reset()
pred = model.predict_generator(test_flow,
                               steps=STEP_SIZE_TEST,
                               verbose=1)
predicted_class_indices = np.argmax(pred, axis=1)
labels = (train_flow.class_indices)
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
filenames = test_flow.filenames
results = pd.DataFrame({"Filename": filenames,
                        "Predictions": predictions})
results.to_csv("results.csv", index=False)

print("Results of predictions on internal test data:")
cp.accuracy_on_generated("results.csv")
print("Results of predictions on external test data:")
