import csv as csv
import numpy as np
from numpy import double


class DataCollector(object):
    """ Klasa pobierania danych

    Parametry
    ------------
    -
    Atrybuty
    -----------
    X : 13 parametrow opisujacych nieruchomosci - na wyjsciu unormowane!
    y: Wartosc nieruchomosci - na wyjsciu unormowana! 
    maxY: Wartosc maksymalna [nieunormowana] nieruchomosci.
    titles: Naglowki kolumn
    """

    def __init__(self):
        self.X=[]
        self.y=[]
        self.maxY=[]
        self.dataFile=open('snbDANE.csv','r')
        dataReader = csv.reader(self.dataFile)
        self.titles = next(dataReader)
        for row in dataReader:
            self.X.append(list(map(double, row[0:len(row) - 1])))
            self.y.append(double(row[len(row) - 1]))
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.standarization()
      

    def standarization(self):
        """Standaryzacja danych i wynikow do zakresu [0;1]

        Parametry
        -----------
        Obiekt klasy

        Zwraca
        ----------
        Znormalizowane dane i wyniki

        """
        for i in range(len(self.X[0])):
            #X_array = np.array(self.X, dtype=double)
            #y_array = np.array(self.y, dtype=double)
            max_value_X = self.X[:, i].max()
            max_value_y = self.y.max()
            for j in range(len(self.X)):
                self.X[j][i] = self.X[j][i] / max_value_X
        self.maxY = max_value_y
        self.y = self.y / max_value_y / 1.05

    def getX(self):
        return self.X


    def getY(self):
        return self.y

    def getHeads(self):
        return self.titles
    
    def getMaxY(self):
        return self.maxY