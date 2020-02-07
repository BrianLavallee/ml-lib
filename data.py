
def load_car():
    train = []
    train_labels = []
    with open("./data/car/train.csv", "r") as f:
        for line in f:
            item = line.strip().split(",")
            train.append(item[:-1])
            train_labels.append(item[-1])

    test = []
    test_labels = []
    with open("./data/car/test.csv", "r") as f:
        for line in f:
            item = line.strip().split(",")
            test.append(item[:-1])
            test_labels.append(item[-1])

    return train, train_labels, test, test_labels
