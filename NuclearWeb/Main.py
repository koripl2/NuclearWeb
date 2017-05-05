from DataCollector import DataCollector
from NeuralWeb import NeuralNetMLP
'''
Created on 20.04.2017

@author: Konrad
'''
def main():
        data_collector = DataCollector()
        net = NeuralNetMLP()
        net.fit(data_collector.X, data_collector.y)

if  __name__ =='__main__':
    main()
