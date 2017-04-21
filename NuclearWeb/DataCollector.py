import csv
'''
Created on 20.04.2017

@author: Konrad
'''
from numpy import double
'''from test.test_zipfile import DATAFILES_DIR'''

class DataCollector(object):
    '''
    classdocs
    '''

    def __init__(self):
        self.x=[]
        self.d=[]
        self.dataFile=open('snbDANE.csv','r')
        self.dataReader = csv.reader(self.dataFile)
        self.titles = next(self.dataReader)
        for row in self.dataReader:
            self.x.append(list(map(double,row[0:len(row)-1])))
            self.d.append(double(row[len(row)-1]))
        self.printData(self.x)
        self.standarization()
        self.findMax()
        print(self.d)
        self.printData(self.x)
    pass

    def findMax(self):
        import numpy as np
        max=[]
        for i in range(len(self.x[0])):
            a = np.array(self.x,dtype=double)
            max.append(a[:,i].max())
        print(max)
    pass

    def printData(self,data):
        for i in range(len(data)):
            print(data[i][:])
    pass

    def standarization(self):
        import numpy as np
        max=[]
        '''print("Dlugosc: " + str(len(self.x[0])))'''
        for i in range(len(self.x[0])):
            '''print("Robie kolumne: " + str(i)) '''
            a = np.array(self.x,dtype=double)
            max = a[:,i].max()
            for j in range(len(self.x)):
                self.x[j][i] = self.x[j][i]/max
        '''print(self.x)'''
    pass

    def getX(self):
        return self.x
    pass

    def getD(self):
        return self.d
    pass

    def getHeads(self):
        return self.titles
    pass

    def load(self):
        import csv

with open("snbDANE.csv", newline='') as csvfile:
    medv = []
    data_in = []
    dataset = csv.reader(csvfile, delimiter=',')
    for row in dataset:
        medv.append(row[13])
        data_in.append(row[0:13])
    print(medv)