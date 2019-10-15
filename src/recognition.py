from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.python.client import device_lib
import pandas as pd
import numpy as np

datagen = ImageDataGenerator()

train_it = datagen.flow_from_directory(
    './data/train/', class_mode='categorical', batch_size=64, target_size=(31, 31))
val_it = datagen.flow_from_directory(
    './data/validate/', class_mode='categorical', batch_size=64, target_size=(31, 31))

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(31, 31, 3),
                 activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same',
                 activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(43, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.fit_generator(train_it, epochs=25,
                    validation_data=val_it, validation_steps=8)


testdf = pd.read_csv(
    "/home/desktop/Desktop/git/traffic_signs/data/test/GT-final_test.csv", dtype=str, delimiter=';')

test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_dataframe(
    dataframe=testdf,
    directory="./data/test/",
    x_col="Filename",
    y_col="ClassId",
    batch_size=32,
    seed=42,
    shuffle=False,
    class_mode="categorical",
    target_size=(31, 31))

    
test_generator.reset()
pred = model.predict_generator(test_generator, verbose=1)

predicted_class_indices = np.argmax(pred, axis=1)

labels = (test_generator.class_indices)
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames = test_generator.filenames
results = pd.DataFrame({"Filename": filenames,
                        "Predictions": predictions})
results.to_csv("results.csv", index=False)
