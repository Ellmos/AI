from neural import *
from dataLoader import GenerateDataSet
from CostFunctions import CostFunctions
from ActivationFunctions import ActivationFunctions

class HyperParameters:
    def __init__(self):
        self.activationFunction = ActivationFunctions.Relu
        self.outputActivationType = ActivationFunctions.Softmax
        self.costFunction = CostFunctions.CrossEntropy
        self.initialLearningRate = 0.025
        self.learnRateDecay = 0.075
        self.batchSize = 16
        self.epoch = 3


if __name__ == "__main__":
    # Create neural network
    hp = HyperParameters()
    neural = NeuralNetwork([784, 30, 10], hp)

    # Create dataSet
    trainDataSet, testDataSet = GenerateDataSet(hp.batchSize)

    trainDataSet = trainDataSet[:100 // hp.batchSize]
    testDataSet[0] = testDataSet[0][:100]

    # Learning
    options = {"debug": True, "graph": False, "saveCSV": False}
    neural.Learn(trainDataSet, testDataSet, hp, options)
