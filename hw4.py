
from data import load_banknote
from SVM.svm import SVM, gamma1, gamma2, SVM_dual

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

    for c in [100/873]:#, 500/873, 700/873]:
        model = SVM_dual(x_train, y_train)
        # model = SVM(x_train, y_train, C=c, gamma=gamma1)
        print(model.w)
        print(error(model, x_train, y_train))
        print(error(model, x_test, y_test))


main()
