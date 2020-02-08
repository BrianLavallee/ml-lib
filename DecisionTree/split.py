
from math import log2

def entropy(labels):
    n = len(labels)

    counts = {}
    for x in labels:
        if x not in counts:
            counts[x] = 0

        counts[x] += 1

    sum = 0
    for x in counts:
        sum += counts[x] / n * log2(n / counts[x])

    return sum

def majority_error(labels):
    n = len(labels)

    majority = 0
    counts = {}
    for x in labels:
        if x not in counts:
            counts[x] = 0

        counts[x] += 1
        if counts[x] > majority:
            majority = counts[x]

    return (n - majority) / n

def gini_index(labels):
    n = len(labels)

    counts = {}
    for x in labels:
        if x not in counts:
            counts[x] = 0

        counts[x] += 1

    sum = 0
    for x in counts:
        sum += (counts[x] / n) ** 2

    return 1 - sum

def gain(attr, train, labels, measure):
    split = {}
    for i in range(len(train)):
        if train[i][attr] not in split:
            split[train[i][attr]] = []

        split[train[i][attr]].append(labels[i])

    g = measure(labels)
    for x in split:
        g -= measure(split[x]) * (len(split[x]) / len(labels))

    return g

def max_gain(train, labels, measure, attributes):
    bestattr = attributes[0]
    bestgain = gain(bestattr, train, labels, measure)
    for attr in attributes[1:]:
        g = gain(attr, train, labels, measure)
        if g > bestgain:
            bestgain = g
            bestattr = attr

    return bestattr
