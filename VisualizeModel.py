from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

class_num = 2
seqLen = 16
timeWindow = 2
features = 14

model = Sequential()
model.add(Conv1D(32, 3, activation='relu', input_shape=(seqLen, features)))
model.add(MaxPooling1D(3))
model.add(Conv1D(64, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(class_num, activation='softmax'))

plot_model(model, to_file='model.png', show_shapes=True)