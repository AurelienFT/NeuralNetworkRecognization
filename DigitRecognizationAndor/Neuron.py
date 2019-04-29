import numpy as np
import numpy.random as random
import Logistic as logistic
import Dataset as dataLoad


class Neuron:
    def __init__(self, nb_entries):
        self.weight = []
        for i in range(0, nb_entries + 1):
            self.weight.append(random.randint(-200, 200))
            self.weight[i] = self.weight[i] / 100
        self.activation = 0
        self.gradient = 0

    def activate(self, entries):
        res = 0
        entries = [1] + entries
        for i in range(0, len(entries)):
            res = res + (entries[i] * self.weight[i])
        self.activation = logistic.f(res)

    def calcLoss(self, dataset):
        res = 0
        for i in range(0, dataset.nb_exemple):
            self.activate(dataset.exemple_list[i].entry)
            res = res + dataset.exemple_list[i].out[0] * np.log(self.activation) + (1 - dataset.exemple_list[i].out[0]) * np.log(1 - self.activation)
        res = -res / dataset.nb_exemple
        return res

    def calculateOutputGradient(self, out):
        self.gradient = out - self.activation

    def applyGradient(self, learningRate, entries):
        entries = [1] + entries
        for i in range(0, len(entries)):
            self.weight[i] = self.weight[i] + learningRate * self.gradient * entries[i]

    def applyGradientWithLayer(self, learningRate, neurons):
        entries = [1] + [i.activation for i in neurons]
        for i in range(0, len(entries)):
            self.weight[i] = self.weight[i] + learningRate * self.gradient * entries[i]

    def calculateHiddenGradient(self, next_layer, index):
        self.gradient = self.activation * (1 - self.activation)
        temp = 0
        for i in range(0, len(next_layer)):
            temp = temp + next_layer[i].gradient * next_layer[i].weight[index + 1]
        self.gradient = self.gradient * temp
