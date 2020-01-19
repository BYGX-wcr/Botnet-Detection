import LoadDataset
import DatasetStatistics
import numpy

def sequentializeDataset(data, labels, batchSize=1000, timeWindow=2, sequenceLen=5, mask_value=0):
    newData = []
    newLabels = []

    counter = 0
    while counter < len(data):
        if sequenceLen != None:
            seqData, seqLabels = extractSequence(data[counter:counter+batchSize], labels[counter:counter+batchSize], timeWindow=timeWindow, sequenceLen=sequenceLen, mask_value=mask_value)
        else:
            seqData, seqLabels = extractVarSequence(data[counter:counter+batchSize], labels[counter:counter+batchSize], timeWindow=timeWindow)
        counter += batchSize
        newData += seqData
        newLabels += seqLabels

    return newData, newLabels

def extractSequence(data, labels, timeWindow, sequenceLen, mask_value):
    """
    This function extract a bunch of record sequences from input data.
    The arg:timeWindow is measured in minutes, arg:sequenceLen indicates each the number of records in each sequence.
    """
    ipMap = dict()
    # aggregate netflow records by source IP, discard MAC record (minority)
    for counter in range(0, len(data)):
        record = data[counter]
        label = labels[counter]
        srcAddr = record[3]

        if srcAddr not in ipMap:
            ipMap[srcAddr] = []
            ipMap[srcAddr].append([record, label])
        else:
            ipMap[srcAddr].append([record, label])

    seqData = []
    seqLabels = []
    # traverse each aggregated record list, extract sequences
    for value in ipMap.values():
        counter = 0
        recList = sorted(value, key=lambda rec: rec[0])
        while counter < len(recList):
            # create sequence based on anchor
            sequence = []
            anchor = recList[counter][0]
            label = recList[counter][1]
            sequence.append(anchor)

            it = 1
            while it < sequenceLen and (counter + it) < len(recList):
                # search for adjacent records
                startTime = recList[counter + it][0][0]
                anchorTime = anchor[0]
                if  startTime - anchorTime < timeWindow * 60:
                    # if the record is within the time window beginning with the anchor, append it
                    sequence.append(recList[counter + it][0])
                    label = max(label, recList[counter + it][1]) # Botnet > Normal > Background
                    it += 1
                else:
                    break

            if it < sequenceLen:
                # if the sequence doesn't contain enough records, append zeros
                deficit = sequenceLen - it
                for i in range(deficit):
                    sequence.append([mask_value] * len(anchor))

            # add sequence into the final sequentialized dataset
            seqData.append(sequence)
            seqLabels.append(label)
            counter += it

    return seqData, seqLabels

def extractVarSequence(data, labels, timeWindow):
    """
    This function extract a bunch of record sequences from input data.
    The arg:timeWindow is measured in minutes, the sequence length is variable
    """
    ipMap = dict()
    # aggregate netflow records by source IP, discard MAC record (minority)
    for counter in range(0, len(data)):
        record = data[counter]
        label = labels[counter]
        srcAddr = record[3]

        if srcAddr not in ipMap:
            ipMap[srcAddr] = []
            ipMap[srcAddr].append([record, label])
        else:
            ipMap[srcAddr].append([record, label])

    seqData = []
    seqLabels = []
    # traverse each aggregated record list, extract sequences
    for value in ipMap.values():
        counter = 0
        recList = sorted(value, key=lambda rec: rec[0])
        while counter < len(recList):
            # create sequence based on anchor
            sequence = []
            anchor = recList[counter][0]
            label = recList[counter][1]
            sequence.append(anchor)

            it = 1
            while (counter + it) < len(recList):
                # search for adjacent records
                startTime = recList[counter + it][0][0]
                anchorTime = anchor[0]
                if  startTime - anchorTime < timeWindow * 60:
                    # if the record is within the time window beginning with the anchor, append it
                    sequence.append(recList[counter + it][0])
                    label = max(label, recList[counter + it][1]) # Botnet > Normal > Background
                    it += 1
                else:
                    break

            # add sequence into the final sequentialized dataset
            sequence = numpy.array(sequence).reshape(len(sequence), len(sequence[0]))
            seqData.append(sequence)
            seqLabels.append(label)
            counter += it

    return seqData, seqLabels

if __name__ == "__main__":
    # Used to test functions defined in this module
    dataset = LoadDataset.Dataset('./CTU-13-Dataset')
    dataset.loadData()
    train_dataset, train_labels, test_dataset, test_labels = dataset.getEntireDataset()

    seqData, seqLabels = sequentializeDataset(train_dataset, train_labels)
    print(len(seqData))
    print(len(seqLabels))
    DatasetStatistics.labelCount(seqLabels)