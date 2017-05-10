import numpy as np
from scipy.special import expit
import sys
import matplotlib.pyplot as plt

class NeuralNetMLP(object):
    """ Siec do przewidywania ceny nieruchomosci na podstawie 13 parametrow

    Parametry
    -----------
    n_output : int
        Liczba neuronow wyjsciowych - 1 (tyle ile wynikow
    n_features : int
        Liczba parametrow wejsciowych
    n_hidden : int (default: 8)
        Liczba neuronow w warstwie ukrytej
    epochs : int (default: 1000)
        Liczba krokow algorytmu uczacego
    eta : float (default: 0.0015)
        Wspolczynnik szybkosci uczenia
    alpha : float (default: 0.0)
        Wspolczynnik bezwladnosci
        w(t) := w(t) - (grad(t) + alpha*grad(t-1)


    Atrybuty
    -----------
    cost_ : list
      Przechowuje srednia wartosc funkcji kosztu w kolejnych iteracjach

    """
    def __init__(self, n_output=1, n_features=13, n_hidden=8,
                 epochs=1000, eta=0.0015, alpha=0.0):
        np.random.seed(0)
        self.n_output = n_output
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.w1, self.w2 = self._initialize_weights()
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha

    def _initialize_weights(self):
        """Inicjalizuje wagi wartosciami losowymi o rozkladzie jednostajnym na przedziala [-1;1]"""

        w1 = np.random.uniform(-1.0, 1.0,self.n_hidden*(self.n_features + 1))#rozkład jednostajny

        w1 = np.random.uniform(-1.0, 1.0,
                               size=self.n_hidden*(self.n_features + 1)) #rozkład jednostajny

        w1 = w1.reshape(self.n_hidden, self.n_features + 1) #z wektora formuje macierz o zadanych ksztaltach
        # w1 jest macierza wag, pierwsza wspolrzedna - do ktorego neuronu z warstwy ukrytej
        #druga - od ktorego wejscia
        w2 = np.random.uniform(-1.0, 1.0,self.n_output*(self.n_hidden + 1))                   
        w2 = w2.reshape(self.n_output, self.n_hidden + 1)
        return w1, w2

    def _sigmoid(self, z):
        """Oblicza wartosc sigmoidalnej funkcji aktywacji neuronu
        Uzywa gotowej implementacji funkcji sigmoidalnej z biblioteki scipy.special
        """
        return expit(z)

    def _sigmoid_gradient(self, z):
        """Oblicza wartosc pochodnej funkcji sigmoidalnej"""
        sg = self._sigmoid(z)
        return sg * (1.0 - sg)

    def _add_bias_unit(self, X, how='column'):
        """Dodaje do wektora wejsciowego kolumne jedynek, w celu dodania polaryzacji,
        kazda z jedynek jest pozniej mnozona przez odpowiednia wage
        """
        #zakladamy, ze wejscie polaryzacji jest zawsze jedynka i od przypisanej wagi zalezy wlasciwa wartosc polaryzacji
        if how == 'column':
            X_new = np.ones((X.shape[0], X.shape[1] + 1))
            X_new[:, 1:] = X
        elif how == 'row':
            X_new = np.ones((X.shape[0] + 1, X.shape[1]))
            X_new[1:, :] = X
        return X_new

    def _feedforward(self, X, w1, w2):
        """Wyznacza wartosc aktywacji kolejnych warst na podstawie zadanego wejscia

        Parametry
        -----------
        X : array, shape = [n_samples, n_features]
            Warstwa wejsciowa
        w1 : array, shape = [n_hidden_units, n_features]
            Wagi polaczen w.wejsciowa-ukryta
        w2 : array, shape = [n_output_units, n_hidden_units]
            Wagi polaczen w.ukryta-wyjsciowa

        Zwraca
        ----------
        a1 : array, shape = [n_samples, n_features+1]
            Wartosci wejscia wraz z polaryzacja
        z2 : array, shape = [n_hidden, n_samples]
            Wejscie warstwy ukrytej
        a2 : array, shape = [n_hidden+1, n_samples]
            Aktywacja warstwy ukrytej - jej wyjscie
        z3 : array, shape = [1, n_samples]
            Wejscie neuronu wyjsciowego
        a3 : array, shape = [n_output_units, n_samples]
            Aktywacja neuronu wyjsciowego - wyjscie ostateczne

        """
        a1 = self._add_bias_unit(X, 'column')
        z2 = w1.dot(a1.T)
        a2 = self._sigmoid(z2)
        a2 = self._add_bias_unit(a2, 'row')
        z3 = w2.dot(a2)
        a3 = self._sigmoid(z3)
        return a1, z2, a2, z3, a3


    def _get_cost(self, d, output, w1, w2):
        """Oblicza wartosc funkcji kosztu korzystajac z funkcji kross entropii

        Parametry
        ---------
        d : array, shape = (1, n_samples)
             Wartosci pozadane dla poszczegolnych wektorow uczacych.
        output : array, shape = (1, n_samples)
             Wartosc na wyjsciu sieci.    
        w1 : array, shape = [n_hidden_units, n_features]
            Wagi od warstwy wejsciowej do ukrytej.
        w2 : array, shape = [n_output_units, n_hidden_units]
            Wagi od warstwy ukrytej do wyjsciowej.


        Zwraca
        --------
        cost : float
         wartosc funkcji kosztu dla wektora

        """

        term1 = -d * (np.log(output)) #cross-entropia lewo
        term2 = (1.0 - d) * np.log(1.0 - output) #cross entropia prawo
        cost = (np.sum(term1 - term2)/len(d)) #blad sredni
        return cost

    def _get_gradient(self, a1, a2, a3, z2, d, w1, w2):
        """ Wyznacza kierunek najwiekszego spadku bledu uzywajac algorytmu propagacji wstecznej

        Parametry
        ------------

        a1 : array, shape = [n_samples, n_features+1]
            Wartosci wejscia wraz z polaryzacja 
        a2 : array, shape = [n_hidden+1, n_samples]
            Wyjscie warstwy ukrytej
        a3 : array, shape = [n_output_units, n_samples]
            Wyjscie neuronu wyjsciowego
        z2 : array, shape = [n_hidden, n_samples]
            Wyjsie warstwy wejsciowej
        z3 : array, shape = [1, n_samples]
            Wejscie neuronu wyjsciowego  
        d : array, shape = (1, n_samples)
             Wartosci pozadane dla poszczegolnych wektorow uczacych.
        w1 : array, shape = [n_hidden_units, n_features]
            Wagi od warstwy wejsciowej do ukrytej.
        w2 : array, shape = [n_output_units, n_hidden_units]
            Wagi od warstwy ukrytej do wyjsciowej.
        Zwraca
        ---------
        delta_w1 : zmiana wag warstwy ukrytej.
        delta_w2 : zmiana wag warstwy wyjsciowej.

        """
        # propagacja wsteczn
        sigma3 = a3 - d
        z2 = self._add_bias_unit(z2, how='row')
        sigma2 = w2.T.dot(sigma3) * self._sigmoid_gradient(z2)
        sigma2 = sigma2[1:, :] #w2.T.dot(sigma3) - blad warstwy ukrytej,
        #pomnozone przez gradient f.akt dla liczenia zmian wag
        #uciety bias
        delta_w1 = self.eta*sigma2.dot(a1) #zmiana wag w.ukrytej
        delta_w2 = self.eta*sigma3.dot(a2.T) #zmiana wag w.wyjsciowej

        return delta_w1, delta_w2


    def predict(self, X):
        """Wyznacza unormowana wartosc ceny nieruchomosci dla zadanego wektora wejsciowego

        Parametry
        -----------

        X : array, shape = [n_samples, n_features]
            Wejscie warsty wejsciowej - 13 wartosci opisujacych nieruchomosc.


        Zwraca:
        ----------

        a3 : array, shape = [n_samples]
            Przewidywana cena.

        """

        a1, z2, a2, z3, a3 = self._feedforward(X, self.w1, self.w2)
        return a3

    def fit(self, X, y, print_progress=False):
        """ Dopasowuje wartosci wag sieci do wprowadzonych danych uczacych.

        Parametry
        -----------

        X : array, shape = [n_samples, n_features]
            Wejscie warsty wejsciowej - 13 wartosci opisujacych nieruchomosc.
        y : array, shape = [n_samples]
            Docelowe wartosci warstwy wyjsciowej.
        print_progress : bool (default: False)
            Drukuje na wyjsciu bledow ile probek zostalo przetworzonych

        ----------
        -

        """
        self.cost_ = []
        X_data, y_data = X.copy(), y.copy()
        delta_w1_prev = np.zeros(self.w1.shape)
        delta_w2_prev = np.zeros(self.w2.shape)

        for i in range(self.epochs):

            if print_progress:
                sys.stderr.write('\rEpoch: %d/%d' % (i+1, self.epochs))
                sys.stderr.flush()
            
            for_vector = np.array_split(range(y_data.shape[0]),1)#od parametru minibatches zalezy czy
            #  uaktualiamy po kazdej probce, czy po grupie probek
            for j in for_vector:
                # Uczymy
                #print(X_data[idx])
                a1, z2, a2, z3, a3 = self._feedforward(X_data[j],self.w1,self.w2)
                                                       
                cost = self._get_cost(y_data[j],a3,self.w1,self.w2)
                                      
                self.cost_.append(cost)

                # Propagacja wsteczna
                delta_w1, delta_w2 = self._get_gradient(a1, a2,a3, z2, y_data[j],self.w1,self.w2)

                #Nowe wagi - z uwzgl�dnieniem bezwladnosci: za *
                self.w1 -= (delta_w1 + (self.alpha * delta_w1_prev))
                self.w2 -= (delta_w2 + (self.alpha * delta_w2_prev))
                #Wartosci do bezwladnosci
                delta_w1_prev, delta_w2_prev = delta_w1, delta_w2
        return self