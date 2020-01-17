import LoadDataset

def sequentializeDataset(data, labels, batchSize=1000, timeWindow=2, sequenceLen=5):
    newData = []
    newLabels = []

    counter = 0
    while counter < len(data):
        seqData, seqLabels = extractSequence(data[counter:counter+batchSize], labels[counter:counter+batchSize], timeWindow=timeWindow, sequenceLen=sequenceLen)
        counter += batchSize
        newData += seqData
        newLabels += seqLabels

    return newData, newLabels

def extractSequence(data, labels, timeWindow, sequenceLen):
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
                    sequence.append([0] * len(anchor))

            # add sequence into the final sequentialized dataset
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

    # test1 = [[19561619.607825,1.026539,1,3440,1577,1,59198,6881,0.8336417106682353,0,0,4,276,156],
    # [19561620.634364,1.009595,1,3440,1577,1,59198,6881,0.8336417106682353,0,0,4,276,156],
    # [19561668.185538,3.056586,1,58712,4768,1,20256,80,3.075853417579041,0,0,3,182,122],
    # [19561668.230897,3.111769,1,58712,4788,1,20256,80,3.075853417579041,0,0,3,182,122],
    # [19561668.963351,3.083411,1,58712,4850,1,20256,80,3.075853417579041,0,0,3,182,122],
    # [19561678.806814,3.097288,1,58712,4866,1,20256,80,3.075853417579041,0,0,3,182,122],
    # [19561894.450457,1.048908,1,59864,47908,1,59198,6881,0.8336417106682353,0,0,4,244,124],
    # [19562095.23132,4.373526,1,15933,1419,1,59198,6881,0.8336417106682353,0,0,4,252,132],
    # [19562233.352114,4.827912,1,15933,1491,1,59198,6881,0.8336417106682353,0,0,4,252,132],
    # [19562323.301515,0.049697,1,37494,41752,1,59364,13363,8.04645138736621,0,0,5,352,208],
    # [19562049.710772,328.361664,1,59198,49185,1,58122,80,6.459909825600065,0,0,7,760,520],
    # [19562434.864769,5.242459,1,15933,1586,1,59198,6881,0.8336417106682353,0,0,4,252,132],
    # [19562476.344485,0.97239,1,28271,28451,1,59198,6881,0.8336417106682353,0,0,4,244,124],
    # [19562779.661695,0.923098,1,28271,13717,1,59198,6881,0.8336417106682353,0,0,4,244,124],
    # [19562861.514293,1.009763,1,35401,1817,1,59198,6881,0.8336417106682353,0,0,4,244,124],
    # [19562898.464075,0.969967,1,38185,42480,1,59198,6881,0.8336417106682353,0,0,4,244,124],
    # [19562976.758829,2.853907,1,1231,2285,1,59364,13363,8.04645138736621,0,0,3,184,122]]
    # test2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    # test1, test2 = sequentializeDataset(test1, test2)