

import numpy
from random import shuffle

class SigmoidLayer:

    def __init__(self, n, prev):
        self.w = numpy.random.normal(0, 1, prev * n)
        self.w.shape = (n, prev)

        self.b = numpy.random.normal(0, 1, n)
        self.b.shape = (n, 1)

    def prop(self, x):
        z = numpy.dot(self.w, x) + self.b
        return 1 / (1 + numpy.exp(-z))

    def backprop(self, x, dz):
        db = dz
        dw = numpy.dot(dz, x.T)
        dz1 = numpy.dot(self.w.T, dz) * x * (1 - x)
        return db, dw, dz1

    def update(self, db, dw):
        self.b -= db
        self.w -= dw

# end of SigmoidLayer

# output layer is always sigmoid
# ind is the dimension of the input
# d is the number of nodes in the hidden layers
# k is the number of hidden layers
# outd is the dimension of the output
class NeuralNet:

    def __init__(self, x, y, width=5):
        self.ind = len(x[0])
        self.d = width
        self.k = 2
        self.outd = 1
        self.layers = []

        prev = self.ind
        for i in range(self.k):
            self.layers.append(SigmoidLayer(self.d, prev))
            prev = self.d

        self.layers.append(SigmoidLayer(self.outd, prev))
        self.train(x, y)

    def prop(self, x):
        x = numpy.array(x)
        x.shape = (self.ind, 1)
        xvals = [x]
        for i in range(self.k + 1):
            x = self.layers[i].prop(x)
            xvals.append(x)

        return xvals

    def backprop(self, xvals, y):
        y = numpy.array(y)
        y.shape = (self.outd, 1)
        dz = xvals[-1] - y
        db = []
        dw = []
        for i in range(self.k, -1, -1):
            b, w, z = self.layers[i].backprop(xvals[i], dz)
            db.insert(0, b)
            dw.insert(0, w)
            dz = z

        return db, dw

    def update(self, db, dw):
        for i in range(self.k + 1):
            self.layers[i].update(db[i], dw[i])

    def train(self, train, label):
        epochs = 100
        batchsize = min(25, len(train))
        lr = .1 / batchsize

        samples = [i for i in range(len(train))]

        for epoch in range(epochs):
            dw = []
            db = []

            shuffle(samples)
            for i in range(batchsize):
                x = train[samples[i]]
                y = label[samples[i]]
                xvals = self.prop(x)
                b, w = self.backprop(xvals, y)
                for j in range(self.k + 1):
                    if j >= len(db):
                        db.append(b[j] * lr)
                        dw.append(w[j] * lr)
                    else:
                        db[j] += b[j] * lr
                        dw[j] += w[j] * lr

            self.update(db, dw)
            print("epoch {}\r".format(epoch), end="")

        print()

    def predict(self, x):
        xvals = self.prop(x)
        return 0 if xvals[-1][0] < 0.5 else 1

# end of NeuralNetClassifier
