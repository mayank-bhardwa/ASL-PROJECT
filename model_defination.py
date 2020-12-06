from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam

input_dims = (32, 32, 1)
n_classes = 29
learning_rate = 0.001
adam = Adam(lr=learning_rate)

classifier = Sequential()

classifier.add(Conv2D(64, (3, 3), padding='same', input_shape=input_dims, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(128, (3, 3), padding='same', input_shape=input_dims, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(256, (3, 3), padding='same', input_shape=input_dims, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(BatchNormalization())

classifier.add(Flatten())
classifier.add(Dropout(0.5))
classifier.add(Dense(1024, activation='sigmoid'))
classifier.add(Dense(n_classes, activation='softmax'))

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
classifier.summary()
