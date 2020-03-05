
from LinearRegression.lms import LinearRegressor, ExactLinearRegressor
from EnsembleLearning.boost import AdaBoost
from EnsembleLearning.bag import BaggedForest
from data import load_concrete, load_bank

import matplotlib.pyplot as plt

def error(model, x, y):
    count = 0
    for i in range(len(x)):
        xi = x[i]
        yi = model.predict(xi)
        if yi != y[i]:
            count += 1

    return count / len(x)

def linear():
    train, train_labels, test, test_labels = load_concrete()

    model = LinearRegressor(train, train_labels)
    print(model.cost(test, test_labels))
    print(model.w)

    x = [i for i in range(len(model.errors))]
    y = model.errors

    plt.plot(x, y)
    plt.ylabel("error")
    plt.xlabel("iterations")
    plt.savefig("figures/lms.png")
    plt.clf()

    model = LinearRegressor(train, train_labels, 1)
    print(model.cost(test, test_labels))
    print(model.w)

    x = [i for i in range(len(model.errors))]
    y = model.errors

    plt.plot(x, y)
    plt.ylabel("error")
    plt.xlabel("iterations")
    plt.savefig("figures/lms_stoc.png")
    plt.clf()

    model = ExactLinearRegressor(train, train_labels)
    print(model.w)

def adaboost():
    train, train_labels, test, test_labels = load_bank()
    model = AdaBoost(train, train_labels)

    x = []
    y1 = []
    y2 = []
    for i in range(1000):
        m = model.stumps[i]
        x.append(i + 1)
        y1.append(error(m, train, train_labels))
        y2.append(error(m, test, test_labels))

    plt.plot(x, y1, label="training")
    plt.plot(x, y2, label="test")
    plt.ylabel("error")
    plt.xlabel("T")
    plt.legend()
    plt.savefig("figures/stump_error.png")
    plt.clf()

    x = []
    y1 = []
    y2 = []
    for i in range(0, 1000, 20):
        model.T = i + 1
        x.append(i + 1)
        y1.append(error(model, train, train_labels))
        y2.append(error(model, test, test_labels))

    plt.plot(x, y1, label="training")
    plt.plot(x, y2, label="test")
    plt.ylabel("error")
    plt.xlabel("T")
    plt.legend()
    plt.savefig("figures/adaboost_error.png")
    plt.clf()

def bagging():
    train, train_labels, test, test_labels = load_bank()
    model = BaggedForest(train, train_labels, 6)

    x = []
    y1 = []
    y2 = []
    for i in range(0, 1000, 20):
        model.T = i + 1
        x.append(i + 1)
        y1.append(error(model, train, train_labels))
        y2.append(error(model, test, test_labels))

    plt.plot(x, y1, label="training")
    plt.plot(x, y2, label="test")
    plt.ylabel("error")
    plt.xlabel("T")
    plt.legend()
    plt.savefig("figures/bagging6_error.png")
    plt.clf()

def main():
    # adaboost()
    # bagging()
    linear()

main()
