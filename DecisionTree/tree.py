
from .split import max_gain, entropy

def count(labels):
    unique = set()
    count = 0
    for x in labels:
        if x not in unique:
            count += 1
            unique.add(x)

    return count

class DecisionTree:

    def __init__(self, train, labels, attributes=None, measure=entropy, max_depth=-1):
        attributes = [i for i in range(len(train[0]))] if attributes == None else attributes
        if count(labels) < 2:
            self.leaf = True
            self.label = labels[0]
            return

        majority = 0
        self.label = -1
        counts = {}
        for x in labels:
            if x not in counts:
                counts[x] = 0

            counts[x] += 1
            if counts[x] > majority:
                majority = counts[x]
                self.label = x

        if max_depth == 0 or len(attributes) == 0:
            self.leaf = True
        else:
            self.leaf = False
            self.attr = max_gain(train, labels, measure, attributes)
            splitt = {}
            splitl = {}
            for i in range(len(train)):
                if train[i][self.attr] not in splitt:
                    splitt[train[i][self.attr]] = []
                    splitl[train[i][self.attr]] = []

                splitt[train[i][self.attr]].append(train[i])
                splitl[train[i][self.attr]].append(labels[i])

            attributes.remove(self.attr)
            self.children = {}
            for v in splitt:
                self.children[v] = DecisionTree(splitt[v], splitl[v], attributes, measure, max_depth - 1)

            attributes.append(self.attr)

    def predict(self, x):
        if self.leaf:
            return self.label

        if x[self.attr] in self.children:
            return self.children[x[self.attr]].predict(x)

        return self.label
