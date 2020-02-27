
from LinearRegression.lms import LinearRegressor, ExactLinearRegressor
from EnsembleLearning.boost import AdaBoost
from EnsembleLearning.bag import BaggedForest
from data import load_concrete, load_bank

def error(model, x, y):
    count = 0
    for i in range(len(x)):
        xi = x[i]
        yi = model.predict(xi)
        if yi != y[i]:
            count += 1

    return count / len(x)

def concrete():
    train, train_labels, test, test_labels = load_concrete()

    model = LinearRegressor(train, train_labels)
    print(model.cost(train, train_labels))
    print(model.cost(test, test_labels))

    model = ExactLinearRegressor(train, train_labels)
    print(model.cost(train, train_labels))
    print(model.cost(test, test_labels))

def bank():
    train, train_labels, test, test_labels = load_bank()
    model = AdaBoost(train, train_labels)
    print(error(model, train, train_labels))
    print(error(model, test, test_labels))
    #
    # print()
    #
    # model = BaggedForest(train, train_labels, 2)
    # print(error(model, train, train_labels))
    # print(error(model, test, test_labels))


def main():
    # concrete()
    bank()

main()
