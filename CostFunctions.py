from math import isnan, log
from enum import Enum
from dataclasses import dataclass


# --------------------------COSTS FUNCTIONS------------------------------- #
def MeanSquare(outputValues, targetValues):
    cost = 0
    for (output, target) in zip(outputValues, targetValues):
        diff = target - output
        cost += diff * diff
    return cost * 0.5

def MeanSquareDerivative(outputValue, targetValue):
    return outputValue - targetValue


def CrossEntropy(outputValues, targetValues):
    cost = 0
    for (output, target) in zip(outputValues, targetValues):
        # output = 1 so plant

        if target == 1:
            tmp = -log(output)
        elif output == 1:
            tmp = 36 # lowest value for python log
        else:
            tmp = -log(1 - output)

        if not isnan(tmp):
            cost += tmp

    return cost


def CrossEntropyDerivative(outputValue, targetValue):
    if outputValue == 0 or outputValue == 1:
        return 0

    return (-outputValue + targetValue) / (outputValue * (outputValue - 1))


@dataclass
class CostFunction:
    function: ()
    derivative: ()


class CostFunctions(Enum):
    MeanSquare = CostFunction(MeanSquare, MeanSquareDerivative)
    CrossEntropy = CostFunction(CrossEntropy, CrossEntropyDerivative)
