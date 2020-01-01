import sys
import keras
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.utils import np_utils

import LoadDataset

if __name__ == "__main__":
    # get training epochs
    epochs = 1
    if len(sys.argv) < 2:
        epochs = int(sys.argv[1])

    # get training model
    model = None
    if len(sys.argv) < 3:
        # create a new model
        model = Sequential()
        model.add(Embedding(14, output_dim=10))
        model.add(LSTM(128))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=keras.losses.categorical_hinge,
                    optimizer=sgd,
                    metrics=['mae', 'accuracy'])
    else:
        # use an existing model
        model = keras.models.load_model(sys.argv[2])

    # get the dataset
    class_num = 3
    dataset = LoadDataset.Dataset("./CTU-13-Dataset")
    dataset.loadData()
    train_dataset, train_labels, test_dataset, test_labels = dataset.getEntireDataset()
    # train_dataset = numpy.array(train_dataset).reshape((len(train_dataset), 14))
    # test_dataset = numpy.array(test_dataset).reshape((len(test_dataset), 14))
    train_labels = np_utils.to_categorical(train_labels, num_classes=class_num, dtype='int')

    model.fit(train_dataset, train_labels, batch_size=512, epochs=epochs)
    model.save("LSTM.model")
    res = model.predict(test_dataset, batch_size=512)
    with open("CNN_predict.result", 'w') as file:
        for label in res:
            index = 0
            for pos in label:
                if pos == 1:
                    file.write(str(index)+"\n")
                index += 1