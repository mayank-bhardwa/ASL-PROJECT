from time import time
import os
import cv2
from tensorflow.keras import utils
from sklearn.model_selection import train_test_split
import numpy as np

train_dir = "asl_alphabet/asl_alphabet_train/asl_alphabet_train/"


def load_data(train_dir):
    images = []
    labels = []
    size = 32, 32
    index = -1
    for folder in os.listdir(train_dir):
        index += 1
        for image in os.listdir(train_dir + "/" + folder):
            temp_img = cv2.imread(train_dir + '/' + folder + '/' + image, 0)
            temp_img = cv2.resize(temp_img, size)
            temp_img = temp_img.reshape(32,32,1)
            images.append(temp_img)
            labels.append(index)

    images = np.array(images)
    images = images.astype('float32') / 255.0
    labels = utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.1)

    print('Loaded', len(x_train), 'images for training,', 'Train data shape =', x_train.shape)
    print('Loaded', len(x_test), 'images for testing', 'Test data shape =', x_test.shape)

    return x_train, x_test, y_train, y_test


start = time()
x_train, x_test, y_train, y_test = load_data(train_dir)
print('Loading:', time() - start)
np.save('x1_train', x_train)
np.save('x1_test', x_test)
np.save('y1_train', y_train)
np.save('y1_test', y_test)
