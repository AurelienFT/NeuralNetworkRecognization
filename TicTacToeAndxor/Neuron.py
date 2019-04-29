#!/bin/python3.6
import numpy as np
import Logistic as Logisti
import Dataset


class Neuron:

    def __init__(self, nbr_entry):
        self.weight = [np.random.randint(-200, 200) / 100.0 for i in range(0, nbr_entry + 1)]
        self.activation = 0
        self.gradient = 0

    def activate(self, entries):
        entries = [1] + entries
        res = 0
        for i in range(0, len(entries)):
            res = res + (self.weight[i] * entries[i])
        self.activation = Logisti.f(res)
        return self.activation

    def calcLoss(self, dataset):

        average = 0.0
        for i in range(0, dataset.nbr_exemple):
            self.activate(dataset.exemple[i].entry)
            out_get = self.activation
            out_wanted = dataset.exemple[i].out[0]
            average += (out_wanted * np.log(out_get) + (1.0 - out_wanted) * np.log((1.0 - out_get)))
        return -average / dataset.nbr_exemple

    def calculateOutputGradient(self, y):
        self.gradient = (y - self.activation)

    def applyGradient(self, learning_rate, entry):
        entry = [1] + entry
        for i in range(0, len(self.weight)):
            self.weight[i] += learning_rate * self.gradient * entry[i]

    def applyGradientWithLayer(self, learning_rate,  neuron):
        entry = [1] + [i.activation for i in neuron]
        for i in range(0, len(self.weight)):
            self.weight[i] += learning_rate * self.gradient * entry[i]

    def calculateHiddenGradient(self, next_layer, index):
        self.gradient = self.activation * (1 - self.activation)
        tot = 0
        for i in range(0, len(next_layer)):
            tot = tot + next_layer[i].gradient * next_layer[i].weight[index + 1]
        self.gradient = self.gradient * tot

