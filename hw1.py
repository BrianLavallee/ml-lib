
from data import load_car, load_bank
from DecisionTree.split import entropy, majority_error, gini_index
from DecisionTree.tree import DecisionTree

def error(dt, x, y):
    count = 0
    for i in range(len(x)):
        xi = x[i]
        yi = dt.predict(xi)
        if yi != y[i]:
            count += 1

    return count / len(x)

def car():
    train, train_labels, test, test_labels = load_car()

    for i in range(1, 7):
        print("maximum depth: {}".format(i))
        dt = DecisionTree(train, train_labels, [i for i in range(len(train[0]))], entropy, i)
        print("train error: {}".format(error(dt, train, train_labels)))
        print("test error: {}".format(error(dt, test, test_labels)))
        print()

def bank():
    train, train_labels, test, test_labels = load_bank()

    for i in range(1, 17):
        print("maximum depth: {}".format(i))
        dt = DecisionTree(train, train_labels, [i for i in range(len(train[0]))], majority_error, i)
        print("train error: {}".format(error(dt, train, train_labels)))
        print("test error: {}".format(error(dt, test, test_labels)))
        print()

def main():
    # car()
    bank()

main()
