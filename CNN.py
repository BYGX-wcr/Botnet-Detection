import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.utils import np_utils

import LoadDataset

if __name__ == "__main__":
    class_num = 3
    dataset = LoadDataset.Dataset("./CTU-13-Dataset")
    dataset.loadData([1, 2])
    train_dataset, train_labels, test_dataset, test_labels = dataset.getShrinkedDataset([1], [2])
    train_labels = np_utils.to_categorical(train_labels, num_classes=class_num, dtype='int')

    model = Sequential()
    model.add(Conv1D(64, 2, activation='relu', input_shape=(None, 14)))
    model.add(Conv1D(64, 2, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(128, 2, activation='relu'))
    model.add(Conv1D(128, 2, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(class_num, activation='sigmoid'))

    model.compile(loss=keras.losses.categorical_hinge,
                optimizer='rmsprop',
                metrics=['mae'])

    model.fit(train_dataset, train_labels, batch_size=512, epochs=10)
    res = model.predict(test_dataset, batch_size=512)
    with open("CNN_predict.result", 'w') as file:
        for label in res:
            file.write(str(label)+"\n")