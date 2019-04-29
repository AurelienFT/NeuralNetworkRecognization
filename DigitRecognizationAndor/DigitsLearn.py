import Dataset as dataLoad
import matplotlib.pyplot as plt
import Neuron
import NeuralNetwork
import numpy as np
import numpy.random as rand

dataset = dataLoad.Dataset("Datasets/trainDigits.ds")
network = NeuralNetwork.NeuralNetwork([64, 10])
i = 0
k = 1
loss = []
iteration = []
plt.ion()

while k > 0.6:
    j = rand.randint(0, dataset.nb_exemple)
    network.activate(dataset.exemple_list[j].entry)
    network.calculateGradients(dataset.exemple_list[j].out)
    network.applyGradients(0.5)
    if i % 30 == 0:
        network.calcLoss(dataset)
        k = network.loss_out
        loss.append(k)
        iteration.append(i)
        print("Iteration " + i.__str__() + ", la valeur de loss est égale à " + loss[int(i / 30)].__str__())
        plt.plot(iteration, loss)
        plt.pause(0.02)
        plt.clf()
    i += 1

dataset = dataLoad.Dataset("Datasets/testDigits.ds")
exepected = []
result = []
for j in range(0, dataset.nb_exemple):
    network.activate(dataset.exemple_list[j].entry)
    exepected.append(dataset.exemple_list[j].out.index(max(dataset.exemple_list[j].out)))
    result.append(network.activation.index(max(network.activation)))
    print(dataset.exemple_list[j].out.index(max(dataset.exemple_list[j].out)).__str__() + " : ", end='')
    print(network.activation.index(max(network.activation)).__str__())

count = 0
for i in range(0, len(exepected)):
    if exepected[i] == result[i]:
        count += 1

count = count / len(exepected) * 100
count = int(count)
print(count.__str__() + "% de reussite")
