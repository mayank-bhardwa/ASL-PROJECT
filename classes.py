import numpy as np
import cv2

x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')
y_test = y_test.argmax(axis=1)
for i in range(0, 29):
    c = 0
    while str(y_test[c]) != str(i):
        c += 1
    im = x_test[c]
    cv2.imshow(str(i), cv2.resize(im, (200, 200)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
