from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
import tensorflow as tf


img_width, img_height = 150, 150

train_data_dir = 'Data-train'
validation_data_dir = 'Data-validation'
epochs = 10
batch_size = 6

train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

opt = tf.keras.optimizers.Adam(
    learning_rate=0.00007,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name='Adam',
)

model = Sequential()
model.add(Conv2D(16, (5, 5), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D((5, 5), padding="same"))

model.add(Conv2D(32, (5, 5), padding="same", activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((5, 5), padding="same"))


model.add(Conv2D(128, (5, 5), padding="same", activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((5, 5), padding="same"))

model.add(Flatten())

model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(4, activation="softmax"))


model.compile(loss='categorical_crossentropy',  optimizer=opt, metrics=['accuracy'])

model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)
