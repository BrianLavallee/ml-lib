
import random
import scipy.optimize
import numpy as np

def shuffle_examples(x, y):
    temp = list(zip(x, y))
    random.shuffle(temp)
    x, y = zip(*temp)
    return x, y

def dot(x, y):
    return sum([x[i] * y[i] for i in range(len(x))])

def gamma1(t):
    return 1 / (1 + 1/1 * t)

def gamma2(t):
    return 1 / (1 + t)

class SVM:
    def __init__(self, x_train, y_train, C=0.1, gamma=gamma1):
        self.w = [0] * (len(x_train[0]) + 1)

        for epoch in range(100):
            x_train, y_train = shuffle_examples(x_train, y_train)

            for i in range(len(x_train)):
                x = x_train[i] + [1]
                if dot(self.w, x) * y_train[i] <= 1:
                    for j in range(len(self.w)):
                        self.w[j] += gamma(i) * C * len(x_train) * y_train[i] * x[j] - gamma(i) * self.w[j]
                else:
                    for j in range(len(self.w) - 1):
                        self.w[j] *= (1 - gamma(i))

    def predict(self, x):
        ext = x + [1]
        pred = 0
        for i in range(len(ext)):
            pred += self.w[i] * ext[i]

        return -1 if pred < 0 else 1

# def dual(x_train, y_train, C):
#     n = len(x_train)
#     k = len(x_train[0])
#     def fun(vars):
#         w = vars[:k]
#         b = vars[k]
#         xi = vars[k+1:k+1+n]
#         a = vars[k+1+n:k+1+2*n]
#         b = vars[k+1+2*n:]
#
#         return w.T @ w + C * np.ones(n).T @ xi - b.T @ xi - a.T @ ((y_train.T * ((x_train @ w) + b * np.ones(n))) - np.ones(n) + xi)
#
#     return fun

def inner_dual(x_train, y_train, C, al, be):
    n = len(x_train)
    k = len(x_train[0])
    def fun(vars):
        w = vars[:k]
        b = vars[k]
        xi = vars[k+1:]

        return w.T @ w + C * np.ones(n).T @ xi - be.T @ xi - al.T @ ((y_train.T * ((x_train @ w) + b * np.ones(n))) - np.ones(n) + xi)

    return fun

def outer_dual(x_train, y_train, C):
    n = len(x_train)
    k = len(x_train[0])
    def fun(vars):
        a = vars[:n]
        b = vars[n:]

        res = scipy.optimize.minimize(inner_dual(x_train, y_train, C, a, b), np.zeros(k+1+n), method="L-BFGS-B")
        return -1 * res.fun

    return fun

class SVM_dual:
    def __init__(self, x_train, y_train, C=0.1):
        x = np.array(x_train)
        y = np.array(y_train)
        fun = outer_dual(x, y, C)

        n = len(x_train)
        k = len(x_train[0])

        lb = [0] * (2*n)
        ub = [np.inf] * (2*n)
        bounds = scipy.optimize.Bounds(lb, ub)

        res = scipy.optimize.minimize(fun, np.zeros(2*n), method="L-BFGS-B", bounds=bounds)
        a = res.x[:n]
        b = res.x[n:]
        res2 = scipy.optimize.minimize(inner_dual(x_train. y_train, C, a, b), np.zeros(k+1+n), method="L-BFGS-B")
        self.w = res2.x[:k+1]

    def predict(self, x):
        ext = x + [1]
        pred = 0
        for i in range(len(ext)):
            pred += self.w[i] * ext[i]

        return -1 if pred < 0 else 1
