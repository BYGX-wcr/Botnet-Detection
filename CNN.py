import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
import sklearn.metrics as sm

import LoadDataset

def prec_score_CNN(y_true, y_pred):
    return sm.precision_score(y_true, y_pred, average='macro')

def f1_score_CNN(y_true, y_pred):
    return sm.f1_score(y_true, y_pred, average='macro')

if __name__ == "__main__":
    dataset = LoadDataset.Dataset("./CTU-13-Dataset")
    dataset.loadData()
    train_dataset, train_labels, test_dataset, test_labels = dataset.getEntireDataset()

    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(None, 14)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=keras.losses.categorical_hinge,
                optimizer='rmsprop',
                metrics=[prec_score_CNN, f1_score_CNN])

    model.fit(train_dataset, train_labels, batch_size=512, epochs=10)
    res = model.predict(test_dataset, batch_size=512)
    with open("CNN_predict.txt", 'w') as file:
        for label in res:
            file.write(str(label)+"\n")