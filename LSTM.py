import sys
import keras
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Input
from keras.layers import LSTM
from keras.utils import np_utils
from imblearn.under_sampling import RandomUnderSampler

import LoadDataset

if __name__ == "__main__":
    # get training epochs
    epochs = 1
    if len(sys.argv) >= 2:
        epochs = int(sys.argv[1])

    # get training model
    model = None
    if len(sys.argv) < 3:
        # create a new model
        model = Sequential()
        model.add(LSTM(32, input_shape=(1, 14)))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['mae', 'accuracy'])
    else:
        # use an existing model
        model = keras.models.load_model(sys.argv[2])

    # get the dataset
    class_num = 3
    class_weights = {0: 0.02, 1: 0.23, 2: 0.75}
    dataset = LoadDataset.Dataset("./CTU-13-Dataset")
    dataset.loadData()
    train_dataset, train_labels, test_dataset, test_labels = dataset.getEntireDataset()

    # list to ndarray
    train_dataset = numpy.array(train_dataset).reshape((len(train_dataset), 1, 14))
    test_dataset = numpy.array(test_dataset).reshape((len(test_dataset), 1, 14))
    train_labels = numpy.array(train_labels)
    test_labels = numpy.array(test_labels)

    if epochs > 0:
        print("Info: Start Training")
        train_labels = np_utils.to_categorical(train_labels, num_classes=class_num, dtype='int') # one-hot encoding
        model.fit(train_dataset, train_labels, batch_size=512, epochs=epochs, class_weight=class_weights)
        model.save("LSTM.model")

    print("Info: Start Testing")
    res = model.predict(test_dataset, batch_size=512)
    print(res)
    with open("LSTM_predict.result", 'w') as file:
        counter = 0
        for prob_vec in res:
            max_class = 0
            max_prob = 0.0
            index = 0
            # find the class with max probability
            for class_prob in prob_vec:
                if class_prob > max_prob:
                    max_class = index
                    max_prob = class_prob
                index += 1

            # output
            file.write(str(max_class) + ',' + str(test_labels[counter]) + '\n')
            counter += 1
