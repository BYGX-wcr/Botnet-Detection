import LoadDataset

def sequentializeDataset(data, labels, batchSize=1000):
    newData = []
    newLabels = []

    counter = 0
    while counter < len(data):
        seqData, seqLabels = extractSequence(data[counter:counter+batchSize], labels[counter:counter+batchSize])
        counter += batchSize
        newData += seqData
        newLabels += seqLabels

    return newData, newLabels

def extractSequence(data, labels, timeWindow=2, sequenceLen=5):
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
        recList = list.sort(value, key=0)
        while counter != len(recList):
            # create sequence based on anchor
            sequence = []
            anchor = recList[counter][0]
            label = recList[counter][1]
            sequence.append(anchor)

            it = 1
            while it < sequenceLen:
                # search for adjacent records
                if recList[counter + it][0] - anchor[0] < timeWindow * 60:
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
                    sequence.append([0] * len(anchor))

            # add sequence into the final sequentialized dataset
            seqData.append(sequence)
            seqLabels.append(label)
            counter += it - 1

    return seqData, seqLabels

if __name__ == "__main__":
    dataset = LoadDataset.Dataset('./CTU-13-Dataset')
    dataset.loadData([1, 2])
    train_dataset, train_labels, test_dataset, test_labels = dataset.getShrinkedDataset([1], [2])

    seqData, seqLabels = sequentializeDataset(train_dataset, train_labels)
    print(len(seqData))
    print(len(seqLabels))