import Neuron as neuron
import numpy as np


class NeuralNetwork:
    def __init__(self, nb_neurons):
        self.loss_out = 0
        self.activation = []
        self.network = []
        self.gradient = 0
        for i in range(0, len(nb_neurons)):
            temp = []
            for j in range(0, nb_neurons[i]):
                if i == 0:
                    temp.append(neuron.Neuron(0))
                else:
                    temp.append(neuron.Neuron(nb_neurons[i - 1]))
            self.network.append(temp)

    def activate(self, entries):
        for i in range(0, len(entries)):
            self.network[0][i].activation = entries[i]
        for i in range(1, len(self.network)):
            temp = []
            for j in range(0, len(self.network[i - 1])):
                temp.append(self.network[i - 1][j].activation)
            for j in range(0, len(self.network[i])):
                self.network[i][j].activate(temp)
            self.activation = [i.activation for i in self.network[-1]]

    def calcLoss(self, dataset):
        self.loss_out = 0
        temp = 0
        for i in range(0, dataset.nb_exemple):
            res = 0
            self.activate(dataset.exemple_list[i].entry)
            for y in range(0, len(self.activation)):
                res = res + dataset.exemple_list[i].out[y] * np.log(self.activation[y]) + (1 - dataset.exemple_list[i].out[y]) * np.log(1 - self.activation[y])
            temp = temp + res
        temp = -temp / dataset.nb_exemple
        self.loss_out = temp

    def calculateGradients(self, out):
        for i in range(0, len(self.network[-1])):
            self.network[-1][i].calculateOutputGradient(out[i])
        for i in reversed(range(0, len(self.network) - 1)):
            for j in range(0, len(self.network[i])):
                self.network[i][j].calculateHiddenGradient(self.network[i + 1], j)

    def applyGradients(self, learningRate):
        for j in range(1, len(self.network)):
            entries = [i for i in self.network[j - 1]]
            for k in self.network[j]:
                k.applyGradientWithLayer(learningRate, entries)
