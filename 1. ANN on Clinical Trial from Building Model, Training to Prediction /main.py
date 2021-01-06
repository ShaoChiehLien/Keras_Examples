import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from os import path

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    # cm is the matrix, 2*2 matrix will have 2*2 pixels
    # interpolation decide if the color between pixels should be blended, nearest means no blended
    # cmap decides the color used to map the scalar data to colors
    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)  # show the title
    plt.colorbar()  # background color represents the value of each indices hold in the matrix

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # normalize only related to printing on console not plotting
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    thresh = cm.max() / 2.  # half of the darkest color
    # iterate through (0, 0), (0, 1), (1, 0), (1, 1)
    # range(cm.shape[0] == range(cm.shape[1]) == 2 in this case
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],  # value cm[i, j] at position [j, i]
                 horizontalalignment="center",
                 #  if the background color is dark(thresh= half of the darkest color), use white text, and vice versa
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()

# ================================================================= #
# Situational Questions and Data
# An experimental drugs was tested on individuals from ages 13 to 100 in a clinical trial
# The trial has 2100 participants. Half were under 65 years old, half were 65 years or older
# 95% of patients 65 or order experienced side effects
# 95% of patients under 65 experienced no side effects
# ================================================================= #
# Create the Training Set
train_samples = []
train_labels = []
for i in range(50):
    # The ~5% of younger individuals who did experience side effects
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(1)

    # The ~5% of older individuals who did not experience side effects
    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(0)

for i in range(1000):
    # The ~95% of younger individuals who did not experience side effects
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(0)

    # The ~95% of older individuals who did experience side effects
    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(1)

# Create the Testing Set
test_labels = []
test_samples = []
for i in range(10):
    # The 5% of younger individuals who did experience side effects
    random_younger = randint(13, 64)
    test_samples.append(random_younger)
    test_labels.append(1)

    # The 5% of older individuals who did not experience side effects
    random_older = randint(65, 100)
    test_samples.append(random_older)
    test_labels.append(0)

for i in range(200):
    # The 95% of younger individuals who did not experience side effects
    random_younger = randint(13, 64)
    test_samples.append(random_younger)
    test_labels.append(0)

    # The 95% of older individuals who did experience side effects
    random_older = randint(65, 100)
    test_samples.append(random_older)
    test_labels.append(1)

# load the previous trained model if exist, and print the optimizer and loss function
if path.exists("medical_trial_model.h5"):
    for i in range(10):
        print()
    print('**************Previous Trained Model ***************')
    previous_trained_model = load_model('medical_trial_model.h5')
    previous_trained_model.summary()
    print('Optimizer: ' + str(previous_trained_model.optimizer))
    print('Loss function: ' + str(previous_trained_model.loss))
    print('**********End of Previous Trained Model ************')
    for i in range(10):
        print()

# ================================================================= #
# ================================================================= #

# Transform the data into the type Tensorflow accept and shuffle the data
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
train_labels, train_samples = shuffle(train_labels, train_samples)

# Normalize and Standardize the training samples
# fit_transform function only accept 2D data, so reshape it from 1*1050 ([, ..., ]) to 1050*1 ([], ..., [])
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1, 1))

# Build a sequential model
model = Sequential([
    Dense(units=16, input_shape=(1, ), activation='relu'),  # input_shape = (batch size, input size)
    Dense(units=32, activation='relu'),  # 32 neurons with activation function 'relu' R(x)= max(0, x)
    # 'softmax' is preferred for single-label classification (ex: either 'dog', 'cat', ....,label couldn't be both)
    # its output would be sum of 1, which is good for representing probability
    Dense(units=2, activation='softmax')
])

model.summary()

# Use Adam Optimizer with learning rate 0.0001 to learn, use sparse_categorical_crossentropy to calculate
# the lose between labels and predictions, use 'accuracy' function to calculate the accuracy, could also use 'mse'
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# validation split = 0.1, split the last 0.1 portion of the training data. Either specifically keep the validation
# data in the last 0.1 portion of the training set, or make sure to shuffle it in case the data is not random
# batch size = 10, calculate the average loss after 1 whole epoch, 1050/10 = 105 iterations and then adjust the weight
# epochs = 30, repeat 30 epochs
# verbose = 2, mention the index of each epochs
model.fit(
    x=scaled_train_samples,
    y=train_labels,
    validation_split=0.1,
    batch_size=10,
    epochs=30,
    verbose=2)

# Transform the data into the type Tensorflow accept and shuffle the data
test_labels = np.array(test_labels)
test_samples = np.array(test_samples)
test_labels, test_samples = shuffle(test_labels, test_samples)
scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1, 1))

# batch size could also increase speed since it's parallel arithmetic
predictions = model.predict(
    x=scaled_test_samples,
    batch_size=10,
    verbose=0
)

# show the predictions in percentages
j = 0
for i in predictions:
    print(i)
    if j == 20:
        break
    j += 1

# return the indices of max values along axis -1, ie, the indices of max value along each row
rounded_predictions = np.argmax(predictions, axis=-1)

# show the indices with the highest percentages
j = 0
for i in rounded_predictions:
    print(i)
    if j == 20:
        break
    j += 1

# Plot a Confusion Matrix
# 2*2 matrix have 4 combination possibilities
# |(False, False) (True, False)|
# |(False, True)  (True, True) |
cm = confusion_matrix(y_true=test_labels, y_pred=rounded_predictions)
cm_plot_labels = ['no_side_effects', 'had_side_effects']

# plot the matrix
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')


# ================================================================= #
# Save the Model
# ================================================================= #

# This method could save all the information, including the architecture, weights,
# optimizer, the state of the optimizer, the learning rate, the loss, etc.
# The state of the optimizer allow to resume training exactly where you left off
# More advanced techniques like how to resume training process won't be included in this example
model.save('medical_trial_model.h5')

# NOTE: It could also be saved in json(only architecture), or saved only weights,
# referred to https://deeplizard.com/learn/video/7n1SpeudvAE for more info
