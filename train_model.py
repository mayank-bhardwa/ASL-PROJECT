from model_defination import classifier
import numpy as np

classifier.summary()

x_train = np.load('x1_train.npy')
y_train = np.load('y1_train.npy')

history = classifier.fit(x_train, y_train, validation_split=0.2, epochs=1, batch_size=64, verbose=1, shuffle=True)

while history.history.get('val_accuracy')[-1] < 0.99:
    history = classifier.fit(x_train, y_train, validation_split=0.2, epochs=1, batch_size=64, verbose=1, shuffle=True)

model_json = classifier.to_json()
with open("model-bw.json", "w") as json_file:
    json_file.write(model_json)
print('Model Saved')
classifier.save_weights('model-bw.h5')
print('Weights saved')
