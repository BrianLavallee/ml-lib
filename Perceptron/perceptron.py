
import random

def shuffle_examples(x, y):
    temp = list(zip(x, y))
    random.shuffle(temp)
    x, y = zip(*temp)
    return x, y

class Perceptron:
    def __init__(self, x_train, y_train):
        self.w = [0] * (len(x_train[0]) + 1)

        for epoch in range(10):
            x_train, y_train = shuffle_examples(x_train, y_train)

            for i in range(len(x_train)):
                x = x_train[i] + [1]
                pred = self.predict(x_train[i])

                if pred != y_train[i]:
                    self.w = [self.w[j] + y_train[i] * x[j] for j in range(len(self.w))]

    def predict(self, x):
        ext = x + [1]
        pred = 0
        for i in range(len(ext)):
            pred += self.w[i] * ext[i]

        return -1 if pred < 0 else 1

class VotingPerceptron:
    def __init__(self, x_train, y_train):
        self.w = [[0] * (len(x_train[0]) + 1)]
        self.c = [0]

        for epoch in range(10):
            x_train, y_train = shuffle_examples(x_train, y_train)
            for i in range(len(x_train)):
                x = x_train[i] + [1]
                pred = self.predict_(self.w[-1], x_train[i])

                if pred != y_train[i]:
                    self.w.append([self.w[-1][j] + y_train[i] * x[j] for j in range(len(self.w[-1]))])
                    self.c.append(0)

                self.c[-1] += 1

        print(len(self.w))

    def predict_(self, w, x):
        ext = x + [1]
        pred = 0
        for i in range(len(ext)):
            pred += w[i] * ext[i]

        return -1 if pred < 0 else 1

    def predict(self, x):
        pred = 0
        for i in range(len(self.w)):
            p = self.predict_(self.w[i], x)
            pred += -self.c[i] if p < 0 else self.c[i]

        return -1 if pred < 0 else 1

class AveragePerceptron:
    def __init__(self, x_train, y_train):
        self.w = [0] * (len(x_train[0]) + 1)
        self.a = [0] * (len(x_train[0]) + 1)

        for epoch in range(10):
            x_train, y_train = shuffle_examples(x_train, y_train)
            for i in range(len(x_train)):
                x = x_train[i] + [1]
                pred = self.predict_(x_train[i])
                if pred != y_train[i]:
                    self.w = [self.w[j] + y_train[i] * x[j] for j in range(len(self.w))]

                self.a = [self.a[j] + self.w[j] for j in range(len(self.w))]

    def predict_(self, x):
        ext = x + [1]
        pred = 0
        for i in range(len(ext)):
            pred += self.w[i] * ext[i]

        return -1 if pred < 0 else 1

    def predict(self, x):
        ext = x + [1]
        pred = 0
        for i in range(len(ext)):
            pred += self.a[i] * ext[i]

        return -1 if pred < 0 else 1
