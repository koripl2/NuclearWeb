

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from DataCollector import DataCollector
from NeuralWeb import NeuralNetMLP
import matplotlib.pyplot as plt
'''
Created on 20.04.2017

@author: Konrad
'''
def main():
        data_collector = DataCollector()

        net = NeuralNetMLP(n_hidden=8, epochs = 2000)
        net.fit(data_collector.X[0:350], data_collector.y[0:350])
        y_calibr = data_collector.y[350:]
        y_net = net.predict(data_collector.X[:][350:])
        _, _, _, _, y_net = net._feedforward(data_collector.X, net.w1, net.w2)
        min_cost = net._get_cost(data_collector.y, data_collector.y, net.w1, net.w2)
        min_cost_arr = np.zeros([net.epochs, 1])
        min_cost_arr.fill(min_cost)
        plt.plot(net.cost_, color='blue')
        plt.plot(min_cost_arr, color='red')
        plt.legend(["osiągnięta wartość", "minimalna wartość"])
        title = "liczba neuronów w warstwie ukrytej = " + str(net.n_hidden) + \
                "\n eta = " + str(net.eta) + " alfa = " + str(net.alpha)
        plt.title(title)
        plt.xlabel('Numer iteracji')
        plt.ylabel('Średnia wartość funkcji kosztu (kross entropii)')
        plt.show()
        plt.plot(y_calibr, color='green')
        plt.plot(y_net[0][350:], color='blue')
        plt.legend(["Wartość dokładna", "Wartość przewidywana"])
        plt.title("Porównanie wartosci na wyjściu sieci z wartościami oczekiwanymi")
        plt.xlabel('Numer wektora')
        plt.ylabel('Wartosc zmiennej objaśniającej')
        plt.show()
if  __name__ =='__main__':
    main()
