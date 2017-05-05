

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

        net = NeuralNetMLP()
        net.fit(data_collector.X[0:350], data_collector.y[0:350])
        y=[]
        y = net.predict(data_collector.X[350:])
        z = data_collector.y[350:]-y
        

        net = NeuralNetMLP( n_hidden=8)
        net.fit(data_collector.X, data_collector.y)
        y_net = np.array(0)
        _, _, _, _, y_net = net._feedforward(data_collector.X, net.w1, net.w2)
        print(" minimal cost value ", net._get_cost(data_collector.y, data_collector.y, net.w1, net.w2) )
        plt.plot(data_collector.y, color='green')
        plt.plot(np.transpose(y_net), color='blue')
        plt.show()
if  __name__ =='__main__':
    main()
