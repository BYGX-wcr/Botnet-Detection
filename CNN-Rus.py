import sys
import keras
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
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
        model.add(Conv1D(32, 2, activation='relu', input_shape=(14, 1)))
        model.add(Conv1D(32, 2, activation='relu'))
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.5))
        model.add(Conv1D(64, 2, activation='relu'))
        model.add(Conv1D(64, 2, activation='relu'))
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(0.5))
        model.add(Dense(3, activation='softmax'))

        sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=keras.losses.categorical_crossentropy,
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

    # conduct undersampling
    rus = RandomUnderSampler(random_state=8)
    train_dataset, train_labels = rus.fit_resample(train_dataset, train_labels)

    # list to ndarray
    train_dataset = numpy.array(train_dataset).reshape((len(train_dataset), 14, 1))
    test_dataset = numpy.array(test_dataset).reshape((len(test_dataset), 14, 1))
    train_labels = numpy.array(train_labels)
    test_labels = numpy.array(test_labels)
    train_labels = np_utils.to_categorical(train_labels, num_classes=class_num, dtype='int')

    if epochs > 0:
        print("Info: Start Training")
        model.fit(train_dataset, train_labels, batch_size=512, epochs=epochs, validation_split=0.2)
        model.save("CNN-Rus.model")

    print("Info: Start Testing")
    res = model.predict(test_dataset, batch_size=512)
    print(res)
    with open("CNN-Rus.result", 'w') as file:
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
