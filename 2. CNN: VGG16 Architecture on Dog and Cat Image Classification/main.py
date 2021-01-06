import numpy as np
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from sklearn.metrics import confusion_matrix

import itertools
import glob
import os
import shutil
import matplotlib.pyplot as plt


def plotImages(images_arr):
    fig, axes = plt.subplots(2, 5, figsize=(20, 3))
    axes = axes.flatten()  # in case the subplots is not one dimension array
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()


# Organize data into train, valid and test directories
os.chdir('dogs-vs-cats')  # change directory to 'dogs-vs-cats', like cd command in terminal
if os.path.isdir('train/dog') is False:
    print('CHECK')
    os.makedirs('train/dog')
    os.makedirs('train/cat')
    os.makedirs('valid/dog')
    os.makedirs('valid/cat')
    os.makedirs('test/dog')
    os.makedirs('test/cat')

    for i in random.sample(glob.glob('cat*'), 500):
        shutil.move(i, 'train/cat')
    for i in random.sample(glob.glob('dog*'), 500):
        shutil.move(i, 'train/dog')
    for i in random.sample(glob.glob('cat*'), 100):
        shutil.move(i, 'valid/cat')
    for i in random.sample(glob.glob('dog*'), 100):
        shutil.move(i, 'valid/dog')
    for i in random.sample(glob.glob('cat*'), 50):
        shutil.move(i, 'test/cat')
    for i in random.sample(glob.glob('dog*'), 50):
        shutil.move(i, 'test/dog')

# get back to the original directory: '2. CNN: VGG16...Image Classification'
os.chdir('../')
print(os.getcwd())

train_path = 'dogs-vs-cats/train'
valid_path = 'dogs-vs-cats/valid'
test_path = 'dogs-vs-cats/test'

# preprocessing_function: vgg16 normalize each pixels' B, G, R by subtract it with the average B, G, R of all pixels
# Target_size: Resize each image to target size (224, 224).
# Classes: Separate the classes to dog and cat based on their directory name
# Batch_size = Process 10 batches(10 images) at a time, and adjust the weights based on the average result
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224, 224), classes=['dog', 'cat'], batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(224, 224), classes=['dog', 'cat'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224, 224), classes=['dog', 'cat'], batch_size=10, shuffle=False)

# Each batch is an array that contains 10 images and their corresponding labels
imgs, labels = next(train_batches)

# Plot the current batch
plotImages(imgs)
print(labels)

# filters: Use x filters, new depth = x * previous depth
# kernel_size: random assign x*y numbers to the x*y matrix and adjust it every time 1 epoch is finished
# activation = 'relu': R(x) = max(0, x)
# padding = 'same': Keep the same size of the image by zero paddings
# pool_size = (x, y): Find the max number in the x*y pool
# strides = x: move the matrix x pixels each time
model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),  # Flatten the 2D or 3D output image to 1D
    Dense(units=2, activation='softmax')  # Two units: cat and dog, show it with probability
])

model.summary()

# Based on the 'categorical_crossentropy' loss function and Adam optimizer to optimize the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# steps_per_epoch: how many steps (batches) it is going to run to consider it as an epoch
# Ex: Data size = 100, if batch size = 5, one full epoch = 20 steps (batches)
# Sometime we want to calculate the loss function and update the weight more often, we reduce the
# steps_per_epoch so it don't need to really run a full epoch(20 steps) but < 20 steps and update
# Note: The next round will start from the batch that was left previously
# More info: https://github.com/keras-team/keras/issues/10164
model.fit(x=train_batches,
          steps_per_epoch=len(train_batches),  # Required in keras
          # len(train_batches) = a full epoch/ batch sizes
          validation_data=valid_batches,
          validation_steps=len(valid_batches),  # Required in keras
          epochs=10,
          verbose=1
)

# ================================================================= #
# Use Testing Set to Predict
# ================================================================= #
# Total number of steps (batches of samples) before declaring the prediction round finished
predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)

# compare the y_true (each sample data's corresponding label) to y_pred
# (the indicies with the highest probability in each row)
cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))

cm_plot_labels = test_batches.class_indices.keys()

# Plot the confusion matrix
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')


# ================================================================= #
# Use Training Set to Predict
# ================================================================= #
# Total number of steps (batches of samples) before declaring the prediction round finished
predictions = model.predict(x=train_batches, steps=len(train_batches), verbose=0)

# compare the y_true (each sample data's corresponding label) to y_pred
# (the indicies with the highest probability in each row)
cm = confusion_matrix(y_true=train_batches.classes, y_pred=np.argmax(predictions, axis=-1))

cm_plot_labels = train_batches.class_indices.keys()

# Plot the confusion matrix
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
