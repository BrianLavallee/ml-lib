
import math
from .tree import WeightedDecisionTree

class AdaBoost:
    def __init__(self, x, y):
        w = [1] * len(x)
        self.T = 1000
        self.votes = []
        self.stumps = []
        for i in range(self.T):
            s = WeightedDecisionTree(x, y, w, max_depth=1)
            self.stumps.append(s)

            error = 0
            for j in range(len(x)):
                if s.predict(x[j]) != y[j]:
                    error += w[j]

            error /= sum(w)

            a = math.log((1 - error) / error) / 2
            self.votes.append(a)

            for j in range(len(w)):
                if y[j] != s.predict(x[j]):
                    w[j] *= math.e ** a
                else:
                    w[j] *= math.e ** -a

    def predict(self, x):
        yes = 0
        no = 0
        for i in range(self.T):
            if self.stumps[i].predict(x) == "yes":
                yes += self.votes[i]
            else:
                no += self.votes[i]

        return "yes" if yes >= no else "no"
