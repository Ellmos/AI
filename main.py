from neural import *
from data.dataLoader import LoadDataSets
from CostFunctions import CostFunctions
from ActivationFunctions import ActivationFunctions

class HyperParameters:
    def __init__(self):
        self.activationFunction = ActivationFunctions.Relu
        self.outputActivationType = ActivationFunctions.Softmax
        self.costFunction = CostFunctions.CrossEntropy
        self.initialLearningRate = 0.025
        self.learnRateDecay = 0.075
        self.batchSize = 0
        self.epoch = 50


if __name__ == "__main__":
    # Create neural network
    hp = HyperParameters()
    neural = NeuralNetwork([784, 300, 10], hp)

    # Create dataSet
    trainDataSet, testDataSet = LoadDataSets(hp.batchSize, mnist=False, modifiedMnist=True, own=True)

    print(len(trainDataSet[0]), len(testDataSet[0]))
    # trainDataSet = trainDataSet[:100 // hp.batchSize]
    # testDataSet[0] = testDataSet[0][:100]

    # Learning
    options = {"debug": True, "graph": True}
    neural.Learn(trainDataSet, testDataSet, hp, options)

