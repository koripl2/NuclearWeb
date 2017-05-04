import sys
import numpy as np
from scipy.special import expit
'''
Created on 20.04.2017

@author: Konrad
'''

class MyClass(object):
    '''
    classdocs
    '''
    
    def __init__(self,x,d,n_output=1,n_input=13,n_hidden=15,
                 eta=0.001,alpha=0.0,random=None):
        np.random.seed(random)
        self.n_output=n_output
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.eta = eta
        self.alpha = alpha
        self.trainingSize = len(x)
        self.x = x
        self.d = d
        self.w1, self.w2 = self._init_weights()
        
    pass

    def _init_weights(self): #wiadomo - losujemy z takim samym pstwem na warto�ci
        w1 = np.random.uniform(-1.0,1.0,size=self.n_input*self.n_hidden)
        w1 = w1.reshape(self.n_hidden,self.n_input) # +1, kt�ry tu nie ma to te biasy, nie u�ywam chwilowo
        #najwy�ej si� te biasy doda p�zniej
        w2 = np.random.uniform(-1.0,1.0,size=self.n_output*self.n_hidden)
        w2 = w2.reshape(self.n_output,self.n_hidden)
        return w1,w2
    pass

    def _sigmoidal_func(self,i): #sigmoidalna do aktywacji
        return expit(i)
    pass

    def _sigmoid_gradient(self,i): #gradient do wstecznej propagacji - gdzie� mu beta zgin�o 
        sig = self._sigmoidal_func(i)
        return sig * (1.0-sig) #tu powinno by� Beta przed pierwszym sig
    pass

    def _forward_prop(self,x,w1,w2):
        z1 = self.x #wej�cie do warstwy nr [1] - wej�ciowej
        z2 = w1.dot(z1.T) #wej�cie do warstwy nr 2 - ukrytej: przemno�enie przez wagi i suma
        a2 = self._sigmoidal_func(z2) #aktywacja sigmoidaln� neuron�w w. ukrytej
        z3 = w2.dot(a2) #wej�cie do warstwy nr 3 - wyj�ciowej: przemno�enie przez wagi i suma
        a3 = self._sigmoidal_func(z3) #aktywacja warstwy wyj�ciowej na potrzeby liczenia b��du entropi�
        return z1, z2, a2, z3, a3
    pass

    def _count_cost(self,d,a3,w1,w2): #liczenie kosztu cross-entropi�
        left = -d * np.log(a3)
        right = (1.0-d)*np.log(1.0-a3)
        cost = np.sum(left - right) #sumujemy, bo zar�wno left, jak i right to wektory, a chcemy jedn� cyfr�
        #we wzorze przed nawiasem sumy jest minus st�d tu te� minus mi�dzy left i right
        return cost
    pass

    def prediction(self,x):
         a1, z2, a2, z3, a3 = self._forward_prop(x, self.w1, self.w2)
         return 
    pass