import numpy as np
import scipy as sp
import sklearn.metrics as sm
from sklearn.naive_bayes import MultinomialNB

import LoadDataset
import Summary

def train_MulNB(data, labels, class_prior=None):
    mnb = MultinomialNB(alpha=2.0, fit_prior=False, class_prior=class_prior)
    mnb.fit(data, labels)

    print("Training of MultinomialNB model finished!")
    return mnb

def predict_MulNB(data, model):
    return model.predict(data)

def experiment(train_dataset, test_dataset, train_labels, test_labels=None):
    MulNB_model = train_MulNB(train_dataset, train_labels)
    res = predict_MulNB(test_dataset, MulNB_model)
    if test_labels != None:
        print("Balanced accuracy: {0}".format(Summary.acc_score(res, test_labels)))
        print("F1 score: {0}".format(Summary.f1_score(res, test_labels)))
        print("Precision score: {0}".format(Summary.prec_score(res, test_labels)))
        print("Recall score: {0}".format(Summary.recall_score(res, test_labels)))

    return res

if __name__ == "__main__":
    dataset = LoadDataset.Dataset("./CTU-13-Dataset")
    dataset.loadData()
    train_dataset, train_labels, test_dataset, test_labels = dataset.getEntireDataset()

    res = experiment(train_dataset, test_dataset, train_labels, test_labels)
    with open("MulNB_predict.result", 'w') as file:
        for label in res:
            file.write(str(label)+"\n")