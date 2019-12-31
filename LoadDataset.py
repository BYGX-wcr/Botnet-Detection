import csv

"""
This python script contain a Dataset class which can load datasets and output in-memory data and labels
"""

class Dataset:
    def __init__(self, path):
        self.filePath = path
        self.data = []
        self.labels = []
        self.range = range(1, 14)
        for i in self.range:
            self.data.append([])
            self.labels.append([])

    def loadData(self, idList=range(1, 14)):
        """Load all SubDatasets specified in the idList"""
        for i in idList:
            with open("{}/{}.csv".format(self.filePath, i), 'r') as file:
                csvReader = csv.reader(file)
                print("Start loading SubDataset #{}".format(i))

                # creat a handler for each field in the vector
                # Table Header: "StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,sTos,dTos,TotPkts,TotBytes,SrcBytes,Label"
                handlers = [float, float, int, int, int, int, int, int, int, int, int, int, int, int, int]
                for line in csvReader:
                    # convert string to corresponding data type
                    vector = []
                    counter = 0
                    for item in line:
                        vector.append(handlers[counter](item))
                        counter += 1

                    # append vector to the SubDataset
                    self.labels[i - 1].append(vector[counter - 1])
                    vector.pop() # pop the label
                    self.data[i - 1].append(vector)

                print("Finish loading SubDataset #{}".format(i))

    def clearCache(self):
        """Clear all loaded SubDatasets"""
        for i in self.range:
            del self.data[i]
            self.data[i] = []
            del self.labels[i]
            self.labels[i] = []

    def getEntireDataset(self):
        """Get the entire dataset with [3, 4, 5, 7, 10, 11, 12, 13] as train dataset and [1, 2, 6, 8, 9] as test dataset"""
        trainData = []
        trainLabels = []
        testData = []
        testLabels = []

        # integrate train dataset
        for i in [3, 4, 5, 7, 10, 11, 12, 13]:
            if len(self.data[i]) == 0:
                print("Warnning: SubDataset #{} hasn't been loaded!".format(i))
            else:
                trainData += self.data[i]
                trainLabels += self.labels[i]

        # integrate test dataset
        for i in [1, 2, 6, 8, 9]:
            if len(self.data[i]) == 0:
                print("Warnning: SubDataset #{} hasn't been loaded!".format(i))
            else:
                testData += self.data[i]
                testLabels += self.labels[i]

        return trainData, trainLabels, testData, testLabels

    def getShrinkedDataset(self, trainIdList, testIdList):
        """Get the shrinked dataset appointed by trainIdList and testIdList"""
        trainData = []
        trainLabels = []
        testData = []
        testLabels = []

        # integrate train dataset
        for i in trainIdList:
            if len(self.data[i]) == 0:
                print("Warnning: SubDataset #{} hasn't been loaded!".format(i))
            else:
                trainData += self.data[i]
                trainLabels += self.labels[i]

        # integrate test dataset
        for i in testIdList:
            if len(self.data[i]) == 0:
                print("Warnning: SubDataset #{} hasn't been loaded!".format(i))
            else:
                testData += self.data[i]
                testLabels += self.labels[i]

        return trainData, trainLabels, testData, testLabels

if __name__ == "__main__":
    dataset = Dataset("./CTU-13-Dataset")
    dataset.loadData([13])
    trainData, trainLabels, testData, testLabels = dataset.getShrinkedDataset([1], [2])
    print("Testing Finished!")
