import sklearn.metrics as sm
import sys
import copy

def acc_score(predict_labels, test_labels):
    return sm.balanced_accuracy_score(test_labels, predict_labels)

def f1_score(predict_labels, test_labels):
    return sm.f1_score(test_labels, predict_labels, average='macro')

def prec_score(predict_labels, test_labels):
    return sm.precision_score(test_labels, predict_labels)

def recall_score(predict_labels, test_labels):
    return sm.recall_score(test_labels, predict_labels)

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("Please specified the file of predicted labels and true labels")
    else:
        predict_labels = []
        test_labels = []
        with open(sys.argv[1], 'r') as file:
            for line in file:
                labels = line.strip().split(',')
                predict_labels.append(int(labels[0]))
                test_labels.append(int(labels[1]))

        # transform multi-class dataset to binary dataset
        t_predict_labels = copy.deepcopy(predict_labels)
        t_test_labels = copy.deepcopy(test_labels)
        for i in range(0, len(predict_labels)):
            if predict_labels[i] == 2:
                t_predict_labels[i] = 1
            else:
                t_predict_labels[i] = 0

            if test_labels[i] == 2:
                t_test_labels[i] = 1
            else:
                t_test_labels[i] = 0

        print("Balanced accuracy: {0}".format(acc_score(predict_labels, test_labels)))
        print("F1 score: {0}".format(f1_score(predict_labels, test_labels)))
        print("Precision score: {0}".format(prec_score(t_predict_labels, t_test_labels)))
        print("Recall score: {0}".format(recall_score(t_predict_labels, t_test_labels)))