
from data import load_banknote
from Perceptron.perceptron import Perceptron, VotingPerceptron, AveragePerceptron

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

    model = Perceptron(x_train, y_train)
    print(model.w)
    print(error(model, x_train, y_train))
    print(error(model, x_test, y_test))
    print()

    model = VotingPerceptron(x_train, y_train)

    # for i in range(len(model.w)):
    #     s = str(model.c[i]) + ": ["
    #     for j in range(len(model.w[i])):
    #         s += "{}".format(round(model.w[i][j], 3)) + ", "
    #
    #     s = s[:-2]
    #     s += "]"
    #     print(s)

    print(error(model, x_train, y_train))
    print(error(model, x_test, y_test))
    print()

    model = AveragePerceptron(x_train, y_train)
    print(model.w)
    print(error(model, x_train, y_train))
    print(error(model, x_test, y_test))

main()
