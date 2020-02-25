
import random
import numpy as np

learning_rate = 0.01

def inner_product(w1, w2):
    sum = 0
    for i in range(len(w1)):
        sum += w1[i] * w2[i]

    return sum

def scalar_product(w, c):
    return [x * c for x in w]

def gradient(train, labels, w):
    grad = [0] * len(w)
    for i in range(len(labels)):
        x = train[i] + [1]
        temp = scalar_product(x, labels[i] - inner_product(w, x))
        grad = [grad[i] - temp[i] for i in range(len(grad))]

    return grad

def sample(train, label, k):
    index = random.sample(range(len(train)), k)
    train_sample = [train[i] for i in range(len(train)) if i in index]
    label_sample = [label[i] for i in range(len(label)) if i in index]
    return train_sample, label_sample

class LinearRegressor:
    def __init__(self, train, labels, batch_size=-1):
        batch_size = len(train) if batch_size < 1 else batch_size
        self.w = [0] * (len(train[0]) + 1)
        self.errors = [self.cost(train, labels)]

        while True:
            batch_x, batch_y = sample(train, labels, batch_size)
            grad = gradient(batch_x, batch_y, self.w)

            oldw = self.w
            temp = scalar_product(grad, learning_rate)
            self.w = [self.w[i] - temp[i] for i in range(len(self.w))]
            self.errors.append(self.cost(train, labels))
            if abs(self.errors[-1] - self.errors[-2]) < 1e-6:
                break

    def predict(self, x):
        x = x + [1]
        return inner_product(self.w, x)

    def cost(self, x, y):
        sum = 0
        for i in range(len(x)):
            sum += (y[i] - self.predict(x[i])) ** 2

        return sum / 2

class ExactLinearRegressor:
    def __init__(self, train, labels):
        t = [x + [1] for x in train]
        x = np.array(t)
        x.shape = (len(t), len(t[0]))
        y = np.array(labels)
        self.w = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)

    def predict(self, x):
        x = x + [1]
        return inner_product(self.w, x)

    def cost(self, x, y):
        sum = 0
        for i in range(len(x)):
            sum += (y[i] - self.predict(x[i])) ** 2

        return sum / 2
