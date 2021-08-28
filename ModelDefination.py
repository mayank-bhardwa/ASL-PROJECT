from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization
from keras.models import Sequential, save_model
from keras.optimizers import adam_v2

input_dims = (100, 100, 1)
n_classes = 27
learning_rate = 0.001
adam = adam_v2.Adam(learning_rate=0.001)
classifier = Sequential()

classifier.add(Conv2D(64, (5, 5), padding='same',
               input_shape=input_dims, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(4, 4)))

classifier.add(Conv2D(128, (5, 5), padding='same',
               input_shape=input_dims, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(4, 4)))

classifier.add(Conv2D(256, (5, 5), padding='same',
               input_shape=input_dims, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(4, 4)))

classifier.add(BatchNormalization())

classifier.add(Flatten())
classifier.add(Dropout(0.5))
classifier.add(Dense(1024, activation='sigmoid'))
classifier.add(Dense(n_classes, activation='softmax'))

classifier.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
