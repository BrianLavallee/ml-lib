
from .tree import WeightedDecisionTree
from random import randint

class BaggedForest:
    def __init__(self, train, labels, sample_size=-1):
        self.trees = []
        for i in range(1000):
            x = []
            y = []
            for j in range(1000):
                index = randint(0, len(train) - 1)
                x.append(train[index])
                y.append(labels[index])

            t = WeightedDecisionTree(x, y, sample_size=sample_size)
            self.trees.append(t)

    def predict(self, x):
        yes = 0
        no = 0
        for t in self.trees:
            if t.predict(x) == "yes":
                yes += 1
            else:
                no += 1

        return "yes" if yes > no else "no"
