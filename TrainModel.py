from ModelDefination import classifier
from LoadData import x_test, x_train, y_test, y_train
import json

classifier.summary()

history = classifier.fit(x_train, y_train, validation_data=(
    x_test, y_test), epochs=10, batch_size=64, verbose=1, shuffle=True)

model_json = classifier.to_json()

with open("model-bw.json", "w") as json_file:
    json_file.write(model_json)
print('Model Saved')

classifier.save_weights('model-bw.h5')
print('Weights saved')

with open('History.json', 'w') as file:
    json.dump(history.history, file)
print("History Saved")
