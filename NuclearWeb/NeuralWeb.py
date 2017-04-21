'''
Created on 20.04.2017

@author: Konrad
'''

class MyClass(object):
    '''
    classdocs
    '''
    outputDim = 1
    inputDim = 13
    epsilon = 0.01
    

    def __init__(self,x,d):
        self.trainingSize = len(x)
        self.x = x
        self.d = d
        
    pass
        