from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Dropout, MaxPooling2D
from keras.callbacks import LambdaCallback
import pre_run
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def prep_batch_callback(batch, logs):
    for image in batch:
        image = pre_run.im_prepare(image)


datagen = ImageDataGenerator()

train_it = datagen.flow_from_directory(
    './data/train/', class_mode='categorical', batch_size=64, target_size=(31, 31))
val_it = datagen.flow_from_directory(
    './data/validate/', class_mode='categorical', batch_size=64, target_size=(31, 31))
test_it = datagen.flow_from_directory(
    './data/test/', class_mode='categorical', batch_size=64, target_size=(31, 31))


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

model.add(Conv2D(128, (3, 3), padding='same',
                 activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(43, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.fit_generator(train_it, epochs=30,
                    validation_data=val_it, validation_steps=8)
loss = model.evaluate_generator(test_it, steps=24)
