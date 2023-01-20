from numpy.random import random

from Perceptron.Amostra import Amostra
import numpy as np

class Perceptron:
    def __init__(self, dados: [Amostra], tempo_treinamento = 1):
        self.dados = [amostra.valor for amostra in dados]
        self._dados: [Amostra] = dados
        self.qtda_atributos = np.array(self.dados).shape[1]
        self.pesos = [ 2 * random() - 1 for i in range(self.qtda_atributos)]
        self.bias = 2 * random() - 1
        self._error = 0
        self._taxa_aprendizado = 1.0e-2
        self._tempo_treinamento = tempo_treinamento
        self._custo = 0
        self._saida = []

    def atualizaCusto(self):
        self._custo += self._error**2

    def calculaError(self, saida_real, saida_perceptron):
        self._error = saida_real - saida_perceptron

    def atualizarBias(self):
        self.bias = self.bias + self._taxa_aprendizado*self._error

    def atualizaPesos(self, amostra: Amostra):
        self.pesos = [peso + self._taxa_aprendizado*self._error*entrada.valor for entrada, peso in zip(amostra.entradas, self.pesos)]

    def fnAtivacao(self, saida_perceptron):
        return 1 if saida_perceptron > 0 else 0

    def step(self, amostra: Amostra):
        calcula_perceptron = sum([entrada.valor * peso for entrada, peso in zip(amostra.entradas, self.pesos)]) + self.bias
        saida_perceptron = self.fnAtivacao(calcula_perceptron)
        self.calculaError(amostra.valor_real, saida_perceptron)
        self.atualizaPesos(amostra)
        self.atualizarBias()
        self.atualizaCusto()
        return saida_perceptron

    def informaAprendizadoACada(self, quantidade, passos):
        if passos % quantidade == 0:
            print('passo {0} : {1} erro(s)'.format(passos, self._custo))

    def informaPeso(self):
        print("Peso: {0}".format(self.pesos))

    def informaBias(self):
        print("Bias: {0}".format(self.bias))

    def informarSaida(self):
        print("saida: {0}".format(self._saida))

    def saida(self):
        self._saida = np.dot(self.dados, np.array(self.pesos)+self.bias)

    def treinar(self):
        for passos in range(self._tempo_treinamento):
            self._custo = 0
            self._saida = [self.step(amostra) for amostra in self._dados]
            self.informaAprendizadoACada(10, passos)
