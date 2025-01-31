
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam

#augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

# preprocess the data
train_set = train_datagen.flow_from_directory('data/train',
                                              target_size=(48, 48),
                                              batch_size=64,
                                              color_mode="grayscale",
                                              class_mode='categorical')

test_set = test_datagen.flow_from_directory('data/test',
                                            target_size=(48, 48),
                                            batch_size=64,
                                            color_mode="grayscale",
                                            class_mode='categorical')

# model architecture
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_set,
                    steps_per_epoch = train_set.n // train_set.batch_size,
                    epochs=50,
                    validation_data=test_set,
                    validation_steps = test_set.n // test_set.batch_size)
model.save('models/emotion_detector.h5')
