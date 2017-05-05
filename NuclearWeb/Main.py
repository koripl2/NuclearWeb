
import numpy as np
import matplotlib.pyplot as plt
from DataCollector import DataCollector
from NeuralWeb import NeuralNetMLP
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
        
if  __name__ =='__main__':
    main()
