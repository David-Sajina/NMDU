import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


dataset_dir = 'Data-train'

img_width, img_height = 300, 300


batch_size = 128
epochs = 4
print("Loading pics...")
def loadPic(file_path):
    img = load_img(file_path, target_size=(img_width, img_height))
    imgArray = img_to_array(img)
    imgArray = imgArray / 255.0
    return imgArray


picDir = []
labels = []

for root, dirs, files in os.walk(dataset_dir):
    for pic in files:
        if pic.endswith('.jpg'):
            fileDir = os.path.join(root, pic)
            label = os.path.basename(root)
            picDir.append(fileDir)
            labels.append(label)

label_to_int = {label: i for i, label in enumerate(set(labels))}
labels = [label_to_int[label] for label in labels]

images = [loadPic(file_path) for file_path in picDir]
labels = tf.keras.utils.to_categorical(labels, num_classes=4)
x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=None)


x_train = np.array(x_train)
x_val = np.array(x_val)
y_train = np.array(y_train)
y_val = np.array(y_val)
print("Loaded!")

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

opt = tf.keras.optimizers.Adam(
    learning_rate=0.0009,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name='Adam',
)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


#def lr_schedule(epoch):
#    if epoch < 6:
#        return 0.0007
#    else:
#        return 0.00001


#lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_schedule)


lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)


history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    validation_data=(x_val, y_val),
    epochs=epochs,
    callbacks=[lr_scheduler]
)

#TEST
test_dataset_dir = 'Data-test'

testDir = []
test_labels = []

for root, dirs, files in os.walk(test_dataset_dir):
    for pic in files:
        if pic.endswith('.jpg'):
            testPath = os.path.join(root, pic)
            test_label = os.path.basename(root)
            testDir.append(testPath)
            test_labels.append(test_label)

test_labels_int = [label_to_int[label] for label in test_labels]
test_labels_categorical = tf.keras.utils.to_categorical(test_labels_int, num_classes=4)

predicted_results = []
for test_image_path, true_label in zip(testDir, test_labels_int):
    test_image_array = loadPic(test_image_path)
    test_image_array = np.array(test_image_array)
    test_image_array = np.expand_dims(test_image_array, axis=0)

    predicted_probs = model.predict(test_image_array)
    predicted_class_idx = np.argmax(predicted_probs)
    predicted_results.append((predicted_probs, predicted_class_idx, true_label))

predicted_classes = [result[1] for result in predicted_results]
true_classes = [result[2] for result in predicted_results]


conf_matrix = confusion_matrix(true_classes, predicted_classes)
graph = ConfusionMatrixDisplay(conf_matrix, display_labels=label_to_int.keys())
graph.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

#plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
#plt.title('Model Accuracy')
#plt.xlabel('Epoch')
#plt.ylabel('Accuracy')
#plt.legend(['Train', 'Validation'], loc='upper left')
#plt.show()

#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('Model Loss')
#plt.xlabel('Epoch')
#plt.ylabel('Loss')
#plt.legend(['Train', 'Validation'], loc='upper left')
#plt.show()

