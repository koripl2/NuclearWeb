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

    def _init_weights(self): #wiadomo - losujemy z takim samym pstwem na wartoœci
        w1 = np.random.uniform(-1.0,1.0,size=self.n_input*self.n_hidden)
        w1 = w1.reshape(self.n_hidden,self.n_input) # +1, który tu nie ma to te biasy, nie u¿ywam chwilowo
        #najwy¿ej siê te biasy doda pózniej
        w2 = np.random.uniform(-1.0,1.0,size=self.n_output*self.n_hidden)
        w2 = w2.reshape(self.n_output,self.n_hidden)
        return w1,w2
    pass

    def _sigmoidal_func(self,i): #sigmoidalna do aktywacji
        return expit(i)
    pass

    def _sigmoid_gradient(self,i): #gradient do wstecznej propagacji - gdzieœ mu beta zginê³o 
        sig = self._sigmoidal_func(i)
        return sig * (1.0-sig) #tu powinno byæ Beta przed pierwszym sig
    pass

    def _forward_prop(self,x,w1,w2):
        z1 = self.x #wejœcie do warstwy nr [1] - wejœciowej
        z2 = w1.dot(z1.T) #wejœcie do warstwy nr 2 - ukrytej: przemno¿enie przez wagi i suma
        a2 = self._sigmoidal_func(z2) #aktywacja sigmoidaln¹ neuronów w. ukrytej
        z3 = w2.dot(a2) #wejœcie do warstwy nr 3 - wyjœciowej: przemno¿enie przez wagi i suma
        a3 = self._sigmoidal_func(z3) #aktywacja warstwy wyjœciowej na potrzeby liczenia b³êdu entropi¹
        return z1, z2, a2, z3, a3
    pass

    def _count_cost(self,d,a3,w1,w2): #liczenie kosztu cross-entropi¹
        left = -d * np.log(a3)
        right = (1.0-d)*np.log(1.0-a3)
        cost = np.sum(left - right) #sumujemy, bo zarówno left, jak i right to wektory, a chcemy jedn¹ cyfrê
        #we wzorze przed nawiasem sumy jest minus st¹d tu te¿ minus miêdzy left i right
        return cost
    pass

    def prediction(self,x):
         a1, z2, a2, z3, a3 = self._forward_prop(x, self.w1, self.w2)
         return 
    pass