import time
import Dataset as dataLoad
import matplotlib.pyplot as plt
import Neuron

dataset = dataLoad.Dataset("Datasets/or.ds")
neuron = Neuron.Neuron(2)
i = 0
loss = []
iteration = []
plt.ion()

while neuron.calcLoss(dataset) > 0.1:
    for j in range(0, dataset.nb_exemple):
        neuron.activate(dataset.exemple_list[j].entry)
        neuron.calculateOutputGradient(dataset.exemple_list[j].out[0])
        neuron.applyGradient(0.5, dataset.exemple_list[j].entry)
    if i % 1 == 0:
        loss.append(neuron.calcLoss(dataset))
        iteration.append(i)
        print("Iteration " + i.__str__() + ", la valeur de loss est égale à " + loss[int(i / 1)].__str__())
        plt.plot(iteration, loss)
        plt.pause(0.02)
        plt.clf()
    i += 1


for j in range(0, dataset.nb_exemple):
    neuron.activate(dataset.exemple_list[j].entry)
    print(dataset.exemple_list[j].entry[0].__str__() + " " + dataset.exemple_list[j].entry[1].__str__() + " " + dataset.exemple_list[j].out[0].__str__() + " : ", end='')
    print("1" if neuron.activation > 0.5 else "0")
