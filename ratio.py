import LoadDataset

if __name__=="__main__":
	dataset = LoadDataset.Dataset('./CTU-13-Dataset')
	dataset.loadData()
	train_dataset, train_labels, test_dataset, test_labels = dataset.getEntireDataset()
	counters = [0, 0, 0]
	for label in train_labels:
		counters[int(label)] += 1
	print(counters)
