
from data import load_banknote
from LogisticRegression.log import LogisticRegression
from NeuralNetworks.nn import NeuralNet

def error(model, x, y):
    count = 0
    for i in range(len(x)):
        xi = x[i]
        yi = model.predict(xi)
        if yi != y[i]:
            count += 1

    return count / len(x)

def main():
    x_train, y_train, x_test, y_test = load_banknote()

    for v in [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]:
        model = LogisticRegression(x_train, y_train, prior=v)
        print(model.w)
        print(error(model, x_train, y_train))
        print(error(model, x_test, y_test))

    print()
    model = LogisticRegression(x_train, y_train, map=False)
    print(model.w)
    print(error(model, x_train, y_train))
    print(error(model, x_test, y_test))
    print()

    for w in [5, 10, 25, 50, 100]:
        model = NeuralNet(x_train, y_train, width=w)
        print(error(model, x_train, y_train))
        print(error(model, x_test, y_test))

main()
