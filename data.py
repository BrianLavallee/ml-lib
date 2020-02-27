
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

def convert_numeric(train, test, attr):
    temp = [x[attr] for x in train]
    temp.sort()
    median = temp[int(len(temp)/2)]
    for x in train:
        x[attr] = 1 if x[attr] > median else 0

    for x in test:
        x[attr] = 1 if x[attr] > median else 0

def convert_categorical(train, test, attr):
    temp = [x[attr] for x in train]
    counts = {}
    majority = 0
    label = -1
    for x in temp:
        if x == "unknown":
            continue

        if x not in counts:
            counts[x] = 0

        counts[x] += 1
        if counts[x] > majority:
            majority = counts[x]
            label = x

    for x in train:
        if x[attr] == "unknown":
            x[attr] = label

    for x in test:
        if x[attr] == "unknown":
            x[attr] = label

def load_bank():
    train = []
    train_labels = []
    with open("./data/bank/train.csv", "r") as f:
        for line in f:
            item = line.strip().split(",")
            train.append(item[:-1])
            train_labels.append(item[-1])

    test = []
    test_labels = []
    with open("./data/bank/test.csv", "r") as f:
        for line in f:
            item = line.strip().split(",")
            test.append(item[:-1])
            test_labels.append(item[-1])

    for attr in [0, 5, 9, 11, 12, 13, 14]:
        convert_numeric(train, test, attr)

    # for attr in [1, 3, 8, 15]:
    #     convert_categorical(train, test, attr)

    return train, train_labels, test, test_labels

def load_concrete():
    train = []
    train_labels = []
    with open("./data/concrete/train.csv", "r") as f:
        for line in f:
            item = line.strip().split(",")
            item = [float(x) for x in item]
            train.append(item[:-1])
            train_labels.append(item[-1])

    test = []
    test_labels = []
    with open("./data/concrete/test.csv", "r") as f:
        for line in f:
            item = line.strip().split(",")
            item = [float(x) for x in item]
            test.append(item[:-1])
            test_labels.append(item[-1])

    return train, train_labels, test, test_labels
