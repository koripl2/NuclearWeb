import csv as csv
import numpy as np
from numpy import double
'''
Created on 20.04.2017

@author: Konrad
from test.test_zipfile import DATAFILES_DIR
'''

class DataCollector(object):
    '''
    classdocs
    '''

    def __init__(self):
        self.X=[]
        self.y=[]
        self.dataFile=open('snbDANE.csv','r')
        dataReader = csv.reader(self.dataFile)
        self.titles = next(dataReader)
        for row in dataReader:
            self.X.append(list(map(double, row[0:len(row) - 1])))
            self.y.append(double(row[len(row) - 1]))
        '''self.printData(self.x)'''
        '''self.findMax()'''
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.standarization()
        '''print(self.d)
        self.printData(self.x)'''

    def findMax(self):
        max=[]
        for i in range(len(self.X[0])):
            a = np.array(self.X, dtype=double)
            max.append(a[:,i].max())
        '''print(max)'''

    def printData(self,data):
        for i in range(len(data)):
            print(data[i][:])

    def standarization(self):
        for i in range(len(self.X[0])):
            #X_array = np.array(self.X, dtype=double)
            #y_array = np.array(self.y, dtype=double)
            max_value_X = self.X[:, i].max()
            max_value_y = self.y.max()
            for j in range(len(self.X)):
                self.X[j][i] = self.X[j][i] / max_value_X
        self.y = self.y / max_value_y

    def getX(self):
        return self.X


    def getY(self):
        return self.y

    def getHeads(self):
        return self.titles