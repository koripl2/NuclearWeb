
import numpy as np
from DataCollector import DataCollector
from NeuralWeb import NeuralNetMLP
import matplotlib.pyplot as plt


def main():

        data_collector = DataCollector()
        net = NeuralNetMLP(n_hidden=8,  epochs =1000, alpha=0.4)
        net.fit(data_collector.X[0:350], data_collector.y[0:350])
        y_calibr = data_collector.y[350:]
        
        y_net = net.predict(data_collector.X[:][350:])
        _, _, _, _, y_net = net._feedforward(data_collector.X, net.w1, net.w2)
        min_cost = net._get_cost(data_collector.y, data_collector.y, net.w1, net.w2)
        min_cost_arr = np.zeros([net.epochs, 1])
        min_cost_arr.fill(min_cost)
        
        plt.plot(net.cost_, color='blue')
        plt.plot(min_cost_arr, color='red')
        plt.ylim([min_cost, max(net.cost_) + 0.04])
        plt.legend(["osiagnieta wartosc", "minimalna wartosc"])
        title = "liczba neuronow w warstwie ukrytej = " + str(net.n_hidden) + \
                "\n eta = " + str(net.eta) + " alfa = " + str(net.alpha)
        plt.title(title)
        plt.xlabel('Numer iteracji')
        plt.ylabel('Srednia wartosc funkcji kosztu (kross entropii)')
        plt.show()
        plt.plot(y_calibr*data_collector.getMaxY(), color='green')
        plt.plot(y_net[0][350:]*data_collector.getMaxY()*1.05, color='blue')
        plt.legend(["Wartosc dokladna", "Wartosc przewidywana"])
        plt.title("Porownanie wartosci na wyjsciu sieci z wartosciami oczekiwanymi")
        plt.xlabel('Numer wektora')
        plt.ylabel('Wartosc')
        plt.show()
     
if  __name__ =='__main__':
    main()
