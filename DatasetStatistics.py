import LoadDataset

def labelCount(labels):
	counters = [0, 0, 0]
	for label in labels:
		counters[int(label)] += 1
	print("Background Normal Botnet")
	print(counters)

if __name__=="__main__":
	dataset = LoadDataset.Dataset('./CTU-13-Dataset')
	dataset.loadData()
	train_dataset, train_labels, test_dataset, test_labels = dataset.getEntireDataset()

	labelCount(train_labels)