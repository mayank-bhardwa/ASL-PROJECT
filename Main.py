import cv2
import numpy as np
from keras.models import model_from_json

classes = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
           'L', 'M', 'N', 'nothing', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V',
           'W', 'X', 'Y', 'Z']
target_size = (100, 100)
target_dims = (1, 100, 100, 1)

with open('./trained model/model-bw.json', 'r') as json_file:
    saved_model = json_file.read()

classifier = model_from_json(saved_model)
classifier.load_weights('trained model/model-bw.h5')

vid = cv2.VideoCapture(0)
frame = vid.read()[1]
screen_size = frame.shape

text_box = np.full(shape=(50, screen_size[1], 3), fill_value=255)
text_box = text_box.astype('uint8')

while True:
    frame = vid.read()[1]
    frame = cv2.rectangle(frame, (50, 50), (250, 250), 2)

    temp = frame[50:250, 50:250]
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    temp = cv2.resize(temp, target_size)
    temp = temp.reshape(target_dims)

    y_predicted = classifier.predict(temp)

    text_box[text_box > 0] = 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (25, 25)
    fontScale = 1
    fontColor = (1, 1, 1)
    lineType = 2

    mx = y_predicted[0].max()
    clss = -1
    for i in range(29):
        if mx == y_predicted[0][i]:
            clss = i
            break

    cv2.putText(text_box, 'Predicted Letter :'+classes[clss],
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

    final_display = cv2.vconcat(src=[frame, text_box])
    cv2.imshow('Sign Language Converter', final_display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

vid.release()
