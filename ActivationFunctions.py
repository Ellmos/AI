from math import sqrt, exp
from enum import Enum
from dataclasses import dataclass
import numpy as np


# --------------------------ACTIVATION FUNCTIONS------------------------------- #
def Relu(weightedSum):
    return weightedSum * (weightedSum > 0)
def ReluDerivative(inputs, index):
    return 0 if inputs[index] < 0 else 1


alpha = 0.05
def LeakyRelu(inputs, index):
    x = inputs[index]
    return x * alpha if x < 0 else x
def LeakyReluDerivative(inputs, index):
    return alpha if inputs[index] < 0 else 1


def Sigmoid(inputs, index):
    return 1 / (1 + exp(-inputs[index]))
def SigmoidDerivative(inputs, index):
    activation = Sigmoid(inputs, index)
    return activation * (1 - activation)


vectExp = np.vectorize(exp)
def Softmax(weightedSum):
    norm = weightedSum - weightedSum.max()
    exps = vectExp(norm)
    sumExp = np.sum(exps)
    return exps / sumExp
def SoftmaxDerivative(weightedSum):
    norm = weightedSum - weightedSum.max()
    exps = vectExp(norm)

    sumExp = np.sum(exps)

    return [[(i*sumExp - i*i) / (sumExp*sumExp)] for i in exps]
    # tmp = exp(inputs[index])
    # return (tmp*sumExp - tmp*tmp) / (sumExp*sumExp)


def Normalize(inputs):
    m = max(inputs)
    for i in range(len(inputs)):
        inputs[i] -= m
    return inputs

def Empty(inputs, index):
    return inputs[index]



# --------------------------WEIGHTS INITIALIZATION------------------------------- #
def HeInitialization(nbrNodesIn, nbrNodesOut):
    std = sqrt(2.0 / nbrNodesIn)
    numbers = np.random.randn(nbrNodesOut, nbrNodesIn)
    return numbers * std



def XavierInitialization(nbrNodesIn, nbrNodesOut):
    upper = 1.0 / sqrt(nbrNodesIn)
    lower = -upper

    numbers = np.random.rand(nbrNodesOut, nbrNodesIn)
    scaled = lower + numbers * (upper - lower)

    return scaled


@dataclass
class ActivationFunction:
    function: ()
    derivative: ()
    weightsInitialization: ()

class ActivationFunctions(Enum):
    Relu = ActivationFunction(Relu, ReluDerivative, HeInitialization)
    LeakyRelu = ActivationFunction(LeakyRelu, LeakyReluDerivative, HeInitialization)
    Sigmoid = ActivationFunction(Sigmoid, SigmoidDerivative, XavierInitialization)
    Softmax = ActivationFunction(Softmax, SoftmaxDerivative, XavierInitialization)
    Empty = ActivationFunction(Empty, Empty, HeInitialization)




