from Perceptron.Entrada import Entrada

class Amostra:
    def __init__(self, valor, _valor_real):
        self._entradas: [Entrada] = valor
        self._valor_real = _valor_real

    @property
    def entradas(self):
        return self._entradas

    @entradas.setter
    def valor(self, valor: [Entrada]):
        self._entradas = valor

    @property
    def valor_real(self):
        return self._valor_real

    @valor_real.setter
    def valor_real(self, valor_real):
        self._valor_real = valor_real