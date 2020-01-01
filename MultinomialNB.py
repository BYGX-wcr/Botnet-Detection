import numpy as np
import scipy as sp
import sklearn.metrics as sm
from sklearn.naive_bayes import MultinomialNB

import LoadDataset

def train_MulNB(data, labels, class_prior=None):
    mnb = MultinomialNB(alpha=2.0, fit_prior=False, class_prior=class_prior)
    mnb.fit(data, labels)

    print("Training of MultinomialNB model finished!")
    return mnb

def predict_MulNB(data, model):
    return model.predict(data)

def score_MulNB(predict_labels, test_labels):
    return sm.balanced_accuracy_score(test_labels, predict_labels)

def f1_score_MulNB(predict_labels, test_labels):
    return sm.f1_score(test_labels, predict_labels, average='macro')

def prec_score_MulNB(predict_labels, test_labels):
    t_predict_labels = predict_labels
    t_test_labels = test_labels
    for i in range(0, len(predict_labels)):
        if predict_labels[i] == 2:
            t_predict_labels[i] = 1
        else:
            t_predict_labels[i] = 0

        if test_labels[i] == 2:
            t_test_labels[i] = 1
        else:
            t_test_labels[i] = 0
    return sm.precision_score(t_test_labels, t_predict_labels)

def experiment(train_dataset, test_dataset, train_labels, test_labels=None):
    MulNB_model = train_MulNB(train_dataset, train_labels)
    res = predict_MulNB(test_dataset, MulNB_model)
    if test_labels != None:
        # print("balanced accuracy: {0}".format(score_MulNB(res, test_labels)))
        print("F1 score: {0}".format(f1_score_MulNB(res, test_labels)))
        print("Precision score: {0}".format(prec_score_MulNB(res, test_labels)))

    return res

if __name__ == "__main__":
    dataset = LoadDataset.Dataset("./CTU-13-Dataset")
    dataset.loadData()
    train_dataset, train_labels, test_dataset, test_labels = dataset.getEntireDataset()

    res = experiment(train_dataset, test_dataset, train_labels, test_labels)
    with open("MulNB_predict.result", 'w') as file:
        for label in res:
            file.write(str(label)+"\n")