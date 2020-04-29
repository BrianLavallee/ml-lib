
import math
import random

def shuffle_examples(x, y):
    temp = list(zip(x, y))
    random.shuffle(temp)
    x, y = zip(*temp)
    return x, y

def dot(x, y):
    return sum([x[i] * y[i] for i in range(len(x))])

def sigmoid(x):
    if x < -700:
        return 0
    return 1 / (1 + math.e ** -x)

def gamma(t):
    return 1 / (1 + t)

class LogisticRegression:
    def __init__(self, x, y, map=True, prior=0.01):
        self.w = [0] * (len(x[0]) + 1)

        c = 0

        for epoch in range(100):
            x, y = shuffle_examples(x, y)

            prev = self.w

            for i in range(len(x)):
                vec = x[i] + [1]
                error = sigmoid(dot(self.w, vec)) - y[i]
                error_vec = [v * error * gamma(c) for v in vec]
                if map:
                    error_vec = [error_vec[j] + self.w[j]/prior*gamma(c) for j in range(len(vec))]

                self.w = [self.w[j] - error_vec[j] for j in range(len(self.w))]
                c += 1

            if dot(self.w, prev) < 1e-5:
                break

    def predict(self, x):
        ext = x + [1]
        pred = sigmoid(dot(self.w, ext))
        return 0 if pred < 0.5 else 1
