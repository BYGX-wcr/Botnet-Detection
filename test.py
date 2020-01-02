import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

res = np.array([[1, 0, 0], [0, 0, 1]])
for label in res:
    index = 0
    for pos in label:
        if pos == 1:
            print(str(index))
        index += 1