"""
# VAMOS ENTENDER OQUE É UM PERCEPTRON

- modelo mais básico de uma neural network (classificador binário linear) and & or
- 1 Neuronio
- N entradas e 1 saída


o perceptron nada mais é que um modelo computacional de um neurônio.
assim como um neurônio, o perceptron terá suas entradas de informações
essas serão mutiplicadas por pesos conhecidos como pesos sinapticos e, somado
a um bias, vai posteriormente passar por uma função de ativação que trará um resutado específico.

 exemplo:
 -> entradas do perceptron
 -> multiplica pelo peso (peso sinapction)
 -> soma-se com um bias
 -> passa pela função de ativação (step)
    => se entrada <= 0, retorna 0
    => se entrada > 0, retorno 1
 -> resultado


 # COMO O PERCEPTRON APRENDE?
 o aprendizado do perceptron acontece com o peso atual somado uma taxa de aprendizagem (conhecido como learning rate) vezes (resutado real - resultado obtido pelo perceptron) multiplicado pela entrada

 Em forma de equação, temos a seguinte regra:
 peso = peso + taxa_aprendizagem * (saida_real - saida_perceptron) * entrada
 bias = bias + taxa_aprendizagem * (saida_real - saida_perceptron) * entrada
"""
from random import random
import numpy as np

from Perceptron.Amostra import Amostra
from Perceptron.Entrada import Entrada
from Perceptron.Perceptron import Perceptron

""""
exercicio 1 : determinar se a pessoa é fã ou não de uma determinada pagina do instagram

As entradas para essa analise será : se o usuário curtiu, compartilhou, comentou e salvou determinada publicação.

"""

dados: [Amostra] = [
    Amostra([Entrada(0), Entrada(0), Entrada(0), Entrada(0)], 0),
    Amostra([Entrada(1), Entrada(0), Entrada(0), Entrada(0)], 0),
    Amostra([Entrada(0), Entrada(1), Entrada(0), Entrada(0)], 0),
    Amostra([Entrada(1), Entrada(1), Entrada(0), Entrada(0)], 1),
    Amostra([Entrada(0), Entrada(0), Entrada(1), Entrada(0)], 0),
    Amostra([Entrada(1), Entrada(0), Entrada(1), Entrada(0)], 1),
    Amostra([Entrada(0), Entrada(0), Entrada(1), Entrada(0)], 0),
    Amostra([Entrada(1), Entrada(0), Entrada(1), Entrada(0)], 1),
    Amostra([Entrada(0), Entrada(1), Entrada(0), Entrada(1)], 1),
    Amostra([Entrada(1), Entrada(1), Entrada(0), Entrada(1)], 1),
    Amostra([Entrada(0), Entrada(0), Entrada(0), Entrada(1)], 0),
    Amostra([Entrada(1), Entrada(0), Entrada(0), Entrada(1)], 1),
    Amostra([Entrada(0), Entrada(0), Entrada(1), Entrada(1)], 0),
    Amostra([Entrada(1), Entrada(0), Entrada(1), Entrada(1)], 1),
    Amostra([Entrada(0), Entrada(1), Entrada(1), Entrada(1)], 1),
    Amostra([Entrada(1), Entrada(1), Entrada(1), Entrada(1)], 1),
]

perceptron = Perceptron(dados, 101)
perceptron.treinar()
perceptron.informaPeso()
perceptron.informaBias()
perceptron.informarSaida()
