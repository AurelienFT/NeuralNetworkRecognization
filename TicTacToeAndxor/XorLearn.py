import Dataset as dataLoad
import matplotlib.pyplot as plt
import Neuron
import NeuralNetwork
import numpy as np
import numpy.random as rand

dataset = dataLoad.Dataset("Datasets/xor.ds")
network = NeuralNetwork.NeuralNetwork([2, 2, 1])
i = 0
k = 1
loss = []
iteration = []
plt.ion()

while k > 0.05:
    for j in range(0, dataset.nb_exemple):
        network.activate(dataset.exemple_list[j].entry)
        network.calculateGradients(dataset.exemple_list[j].out)
        network.applyGradients(0.1)
    if i % 500 == 0:
        network.calcLoss(dataset)
        k = network.loss_out
        loss.append(k)
        iteration.append(i)
        print("Iteration " + i.__str__() + ", la valeur de loss est égale à " + loss[int(i / 500)].__str__())
        plt.plot(iteration, loss)
        plt.pause(0.02)
        plt.clf()
    i += 1

dataset = dataLoad.Dataset("Datasets/xor.ds")
exepected = []
result = []
for j in range(0, dataset.nb_exemple):
    network.activate(dataset.exemple_list[j].entry)
    exepected.append(dataset.exemple_list[j].out[0])
    result.append(1 if network.activation[0] > 0.5 else 0)
    print(dataset.exemple_list[j].entry[0].__str__() + " ", end='')
    print(dataset.exemple_list[j].entry[1].__str__() + " ", end='')
    print(dataset.exemple_list[j].out[0].__str__() + " : ", end='')
    print("1" if network.activation[0] > 0.5 else "0")

count = 0
for i in range(0, len(exepected)):
    if exepected[i] == result[i]:
        count += 1
plt.plot(iteration, loss)
plt.show()
count = count / len(exepected) * 100
count = count
print(count.__str__() + "% de reussite")
