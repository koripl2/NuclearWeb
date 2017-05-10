import numpy as np
from scipy.special import expit
import sys
import matplotlib.pyplot as plt

'''
Based on https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch12/ch12.ipynb
'''
class NeuralNetMLP(object):
    """ Siec do przewidywania ceny nieruchomosci na podstawie 13 parametrow

    Parametry
    ------------
    n_output : int
        Liczba neuronow wyjsciowych - 1 (tyle ile zmiennych objasniajacych
    n_features : int
        Liczba parametrow wejsciowych
    n_hidden : int (default: 30)
        Liczba neuronow w warstwie ukrytej
    epochs : int (default: 1000)
        Liczba krokow algorytmu uczacego
    eta : float (default: 0.001)
        Wspolczynnik szybkosci uczenia
    alpha : float (default: 0.0)
        Wspolczynnik bezwladnosci
        w(t) := w(t) - (grad(t) + alpha*grad(t-1))
    minibatches : int (default: 1)
        Divides training data into k minibatches for efficiency.
    random_state : int (default: None)
        Uzywany do inicjalizowania wag losowymi wartosciami

    Atrybuty
    -----------
    cost_ : list
      Przechowuje srednia wartosc funkcji kosztu w kolejnych iteracjach

    """
    def __init__(self, n_output=1, n_features=13, n_hidden=8,
                 epochs=1000, eta=0.0012, alpha=0.0, shuffle=True,
                 minibatches=1, random_state=None):

        np.random.seed(random_state)
        self.n_output = n_output
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.w1, self.w2 = self._initialize_weights()
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha
        self.shuffle = shuffle
        self.minibatches = minibatches


    def _initialize_weights(self):
        """Inicjalizuje wagi wartosciami losowymi o rozkladzie jednostajnym na przedziala [-1;1]"""
        w1 = np.random.uniform(-1.0, 1.0,
                               size=self.n_hidden*(self.n_features + 1)) #rozkład jednostajny
        w1 = w1.reshape(self.n_hidden, self.n_features + 1) #z wektora formuje macierz o zadanych ksztaltach
        # w1 jest macierza wag, pierwsza wspolrzedna - do ktorego neuronu z warstwy ukrytej
        #druga - od ktorego wejscia
        w2 = np.random.uniform(-1.0, 1.0,
                               size=self.n_output*(self.n_hidden + 1))
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
        else:
            raise AttributeError('`how` must be `column` or `row`')
        return X_new

    def _feedforward(self, X, w1, w2):
        """Wyznacza wartosc aktywacji kolejnych warst na podstawie zadanego wejscia

        Parametry
        -----------
        X : array, shape = [n_samples, n_features]
            Warstwa wejsciowa
        w1 : array, shape = [n_hidden_units, n_features]
            Wagi polaczen warstwy wejsciowej
        w2 : array, shape = [n_output_units, n_hidden_units]
            Wagi polaczen warstwy ukrytej

        Zwraca
        ----------
        a1 : array, shape = [n_samples, n_features+1]
            Wartosci wejscia wraz z polaryzacja
        z2 : array, shape = [n_hidden, n_samples]
            Wyjsie warstwy wejsciowej
        a2 : array, shape = [n_hidden+1, n_samples]
            Wyjscie warstwy ukrytej
        z3 : array, shape = [1, n_samples]
            Wejscie neuronu wyjsciowego
        a3 : array, shape = [n_output_units, n_samples]
            Wyjscie neuronu wyjsciowego

        """
        a1 = self._add_bias_unit(X, how='column')
        z2 = w1.dot(a1.T)
        a2 = self._sigmoid(z2)
        a2 = self._add_bias_unit(a2, how='row')
        z3 = w2.dot(a2)
        a3 = self._sigmoid(z3)
        return a1, z2, a2, z3, a3


    def _get_cost(self, d, output, w1, w2):
        """Oblicza wartosc funkcji kosztu korzystajac z funkcji kross entropii

        Parametry
        ----------
        d : array, shape = (1, n_samples)
             Wartosci pozadane dla poszczegolnych wektorow uczacych.
        output : array, shape = (1, n_samples)
             Wartosc na wyjsciu.    
        w1 : array, shape = [n_hidden_units, n_features]
            Wagi od warstwy wejsciowej do ukrytej.
        w2 : array, shape = [n_output_units, n_hidden_units]
            Wagi od warstwy ukrytej do wyjsciowej.

        Zwraca
        ---------
        cost : float

        """
        term1 = -d * (np.log(output))
        term2 = (1.0 - d) * np.log(1.0 - output)
        cost = (np.sum(term1 - term2)/len(d))
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
        grad1 : array, shape = [n_hidden_units, n_features]
            Gradient of the weight matrix w1.
        grad2 : array, shape = [n_output_units, n_hidden_units]
            Gradient of the weight matrix w2.

        """
        # propagacja wsteczna
        sigma3 = a3 - d
        z2 = self._add_bias_unit(z2, how='row')
        sigma2 = w2.T.dot(sigma3) * self._sigmoid_gradient(z2)
        sigma2 = sigma2[1:, :]
        grad1 = sigma2.dot(a1)
        grad2 = sigma3.dot(a2.T)

        return grad1, grad2

    def predict(self, X):
        """Wyznacza unormowana wartosc ceny nieruchomosci dla zadanego wektora wejsciowego

        Parametry
        -----------
        X : array, shape = [n_samples, n_features]
            Wejscie warsty wejsciowej - 13 wartosci opisujacych nieruchomosc.

        Zwraca:
        ----------
        y_pred : array, shape = [n_samples]
            Przewidywana cena.

        """
        if len(X.shape) != 2:
            raise AttributeError('X must be a [n_samples, n_features] array.\n'
                                 'Use X[:,None] for 1-feature classification,'
                                 '\nor X[[i]] for 1-sample classification')

        a1, z2, a2, z3, a3 = self._feedforward(X, self.w1, self.w2)
        y_pred = a3
        return y_pred

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

        Zwraca:
        ----------
        self

        """
        self.cost_ = []
        X_data, y_data = X.copy(), y.copy()
        delta_w1_prev = np.zeros(self.w1.shape)
        delta_w2_prev = np.zeros(self.w2.shape)

        for i in range(self.epochs):

            if print_progress:
                sys.stderr.write('\rEpoch: %d/%d' % (i+1, self.epochs))
                sys.stderr.flush()

            if self.shuffle:
                idx = np.random.permutation(y_data.shape[0]) #wybiera losowy wiersz
                X_data, y_data = X_data[idx], y_data[idx] # [idx] - wybiera wylosowany wiersz,
                #  [:,x] - wybiera wylosowana kolumne
            
            mini = np.array_split(range(y_data.shape[0]), self.minibatches)#od parametru minibatches zalezy czy
            #  uaktualiamy po kazdej probce, czy po grupie probek
            for idx in mini:

                # feedforward
                #print(X_data[idx])
                a1, z2, a2, z3, a3 = self._feedforward(X_data[idx],
                                                       self.w1,
                                                       self.w2)
                cost = self._get_cost(d=y_data[idx],
                                      output=a3,
                                      w1=self.w1,
                                      w2=self.w2)
                self.cost_.append(cost)

                # compute gradient via backpropagation
                grad1, grad2 = self._get_gradient(a1=a1, a2=a2,
                                                  a3=a3, z2=z2,
                                                  d=y_data[idx],
                                                  w1=self.w1,
                                                  w2=self.w2)

                delta_w1, delta_w2 = self.eta * grad1, self.eta * grad2
                self.w1 -= (delta_w1 + (self.alpha * delta_w1_prev))
                self.w2 -= (delta_w2 + (self.alpha * delta_w2_prev))
                delta_w1_prev, delta_w2_prev = delta_w1, delta_w2
        return self