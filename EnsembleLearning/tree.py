
from math import log2
from random import sample

def weighted_entropy(y, w):
    counts = {}
    for i in range(len(y)):
        if y[i] not in counts:
            counts[y[i]] = 0

        counts[y[i]] += w[i]

    n = sum(w)
    ent = 0
    for x in counts:
        ent -= counts[x] / n * log2(counts[x] / n)

    return ent

def gain(attr, x, y, w):
    splity = {}
    splitw = {}
    for i in range(len(x)):
        val = x[i][attr]
        if val not in splity:
            splity[val] = []
            splitw[val] = []

        splity[val].append(y[i])
        splitw[val].append(w[i])

    g = weighted_entropy(y, w)
    n = sum(w)
    for val in splity:
        g -= weighted_entropy(splity[val], splitw[val]) * sum(splitw[val]) / n

    return g

def max_gain(x, y, w, attributes, sample_size):
    poss = sample(attributes, sample_size)
    bestattr = poss[0]
    bestgain = gain(bestattr, x, y, w)
    for attr in poss[1:]:
        g = gain(attr, x, y, w)
        if g > bestgain:
            bestgain = g
            bestattr = attr

    return bestattr

class WeightedDecisionTree:
    def __init__(self, x, y, w=None, attributes=None, max_depth=-1, sample_size=-1):
        if attributes == None:
            attributes = [i for i in range(len(x[0]))]

        if w == None:
            w = [1] * len(x)

        if sample_size < len(attributes):
            sample_size = len(attributes)

        counts = {}
        majority = 0
        self.label = 0
        for i in range(len(y)):
            if y[i] not in counts:
                counts[y[i]] = 0

            counts[y[i]] += w[i]
            if counts[y[i]] > majority:
                majority = counts[y[i]]
                self.label = y[i]

        if len(counts) < 2 or max_depth == 0 or len(attributes) == 0:
            self.leaf = True
            return

        self.leaf = False
        self.attr = max_gain(x, y, w, attributes, sample_size)
        splitx = {}
        splity = {}
        splitw = {}
        for i in range(len(x)):
            val = x[i][self.attr]
            if val not in splitx:
                splitx[val] = []
                splity[val] = []
                splitw[val] = []

            splitx[val].append(x[i])
            splity[val].append(y[i])
            splitw[val].append(w[i])

        attributes.remove(self.attr)
        self.children = {}
        for val in splitx:
            self.children[val] = WeightedDecisionTree(splitx[val], splity[val], splitw[val], attributes, max_depth - 1)

        attributes.append(self.attr)

    def predict(self, x):
        if self.leaf:
            return self.label

        if x[self.attr] in self.children:
            return self.children[x[self.attr]].predict(x)

        return self.label
