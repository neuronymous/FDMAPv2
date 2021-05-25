from collections import OrderedDict
from random import shuffle

from numpy import random


class Distribution:
    precision = 3
    epsilon = 10e-3

    def __init__(self, values_to_weights):
        weights = values_to_weights.values()
        s = sum(weights)
        # highest to lowest
        sorted_values_to_weights = sorted(values_to_weights.items(), key=lambda item: item[1], reverse=True)
        self.dist = {v: round(w / s, Distribution.precision) for v, w in
                     sorted_values_to_weights}
        if sum(self.dist.values()) < 1 - Distribution.epsilon:
            raise Exception("Distribution doesn't sum up to 1")

    def sample(self, random_token=None):
        if random_token is not None:
            used_prob = 1.0
            inverse_dist = OrderedDict()  # we add from highest to lowest
            for p in self.dist.values():
                for v in self.dist.keys():
                    if self.dist[v] == p:
                        if p not in inverse_dist:
                            inverse_dist[p] = []
                        inverse_dist[p].append(v)
            for p, values in inverse_dist.items():
                shuffle(values)  # shuffle values with identical probabilities
                for v in values:
                    used_prob -= p
                    if used_prob <= random_token:
                        return v
            raise Exception("Shouldn't get here, didn't manage to sample anything")

        else:
            elements = list(self.dist.keys())
            probabilities = [self.dist[elem] for elem in elements]
            return random.choice(elements, 1, p=probabilities)[0]
