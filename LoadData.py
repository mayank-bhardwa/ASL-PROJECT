import os
import cv2
import numpy as np
from tensorflow.keras import utils
from sklearn.model_selection import train_test_split

train_dir = "data set/generated images/"


def load_data(train_dir):
    images = []
    labels = []
    size = 100, 100
    index = -1
    for folder in os.listdir(train_dir):
        index += 1
        for image in os.listdir(train_dir + "/" + folder):
            temp_img = cv2.imread(train_dir + '/' + folder + '/' + image, 0)
            temp_img = cv2.resize(temp_img, size)
            temp_img = temp_img.reshape(100, 100, 1)
            images.append(temp_img)
            labels.append(index)

    images = np.array(images)
    images = images.astype('float32') / 255.0
    labels = utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.1)

    print('Loaded', len(x_train), 'images for training,',
          'Train data shape =', x_train.shape)
    print('Loaded', len(x_test), 'images for testing',
          'Test data shape =', x_test.shape)

    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = load_data(train_dir)
print('Loaded Data')
