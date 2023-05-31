from ActivationFunctions import ActivationFunctions, Normalize
import json
from math import exp
import numpy as np

import matplotlib.pyplot as plt
from random import shuffle
from time import time


def NeuralFromJson(filePath, hyperParameters):
    with open(filePath, 'r') as file:
        content = json.loads(file.read())
        layersSizes = content["layersSizes"]
        neural = NeuralNetwork(layersSizes, hyperParameters)

        for (layer, layerData) in zip(neural.layers, content["layers"]):
            layer.weights = layerData["weights"]
            layer.biases = layerData["biases"]

    return neural

class NeuralNetwork:
    def __init__(self, layersSizes, hyperParameters):
        self.nbrLayers = len(layersSizes) - 1
        self.layers = [Layer(layersSizes[i], layersSizes[i + 1], hyperParameters.activationFunction) for i in range(self.nbrLayers)]
        self.layers[-1].SetActivationFunction(hyperParameters.outputActivationType)
        self.Cost = hyperParameters.costFunction.value.function
        self.CostDerivative = hyperParameters.costFunction.value.derivative

    def ToJson(self, path):
        jsonObject = {
            "layersSizes": [self.layers[0].nbrNodesIn],
            "layers": []
        }

        for layer in self.layers:
            jsonObject["layersSizes"].append(layer.nbrNodesOut)
            jsonObject["layers"].append(layer.ToJson())

        with open(f"saves/{path}.json", "w") as save:
            save.write(json.dumps(jsonObject))

    def SetActivationFunctions(self, ActivationFunction, outputActivationFunction):
        for layer in self.layers:
            layer.SetActivationFunction(ActivationFunction)

        self.layers[-1].SetActivationFunction(outputActivationFunction)

    def SetCostFunction(self, CostFunction):
        self.Cost = CostFunction.value.function
        self.CostDerivative = CostFunction.value.derivative


    def CalculateOutputs(self, inputs):
        for layer in self.layers:
            inputs = layer.CalculateOutputs(inputs)

        return inputs

    def FeedBatch(self, batch, learningRate):
        for dataPoint in batch:
            self.CalculateOutputs(dataPoint.input)

            outputLayer = self.layers[-1]
            previousOutputs = self.layers[-2].outputs if self.nbrLayers >= 2 else dataPoint.input

            # Compute the nodes values of the output layer and update its gradients
            nodesValues = []
            if outputLayer.ActivationDerivative == ActivationFunctions.Softmax.value.derivative:
                activationDerivatives = outputLayer.ActivationDerivative(outputLayer.weightedSum)
                costDerivatives = self.CostDerivative(outputLayer.outputs, dataPoint.target)

            for nodesOut in range(outputLayer.nbrNodesOut):
                activationDerivative = outputLayer.ActivationDerivative(outputLayer.weightedSum, nodesOut)
                costDerivative = self.CostDerivative(outputLayer.outputs[nodesOut], dataPoint.target[nodesOut])
                currentNodeValue = costDerivative * activationDerivative
                nodesValues.append(currentNodeValue)

                outputLayer.gradientBiases[nodesOut] += currentNodeValue
                for nodesIn in range(outputLayer.nbrNodesIn):
                    outputLayer.gradientWeights[nodesOut][nodesIn] += previousOutputs[nodesIn] * currentNodeValue


            # Go back through the layers, compute the corresponding node values and update the gradient at the same time
            for i in range(2, self.nbrLayers + 1):
                previousOutputs = self.layers[-i - 1].outputs if i < self.nbrLayers else dataPoint.input

                currentLayer = self.layers[-i]
                nodesValues = currentLayer.UpdateGradient(self.layers[-i + 1], nodesValues, previousOutputs)

        for layer in self.layers:
            layer.ApplyGradient(learningRate / len(batch))



    def Learn(self, trainDataSet, testDataSet, hp, options):

        nbrBatch = len(trainDataSet)
        printBatch = nbrBatch // 10 if nbrBatch // 10 != 0 else 1

        accuracyTrain = []
        accuracyValidation = []

        t = time()
        print("\n------------------Learning-----------------------", end="")
        for currentEpoch in range(hp.epoch):
            if options["debug"]:
                print("\n--Epoch {} out of {}--".format(currentEpoch + 1, hp.epoch))

            learningRate = hp.initialLearningRate * (1 / (1 + hp.learnRateDecay * currentEpoch))
            shuffle(trainDataSet)
            for i, batch in enumerate(trainDataSet):
                if options["debug"] and i % printBatch == 0:
                    print("Batch {} out of {}".format(i, nbrBatch))

                self.FeedBatch(batch, learningRate)

            accuracyTrain.append(self.DataSetAccuracy(trainDataSet))
            accuracyValidation.append(self.DataSetAccuracy(testDataSet))


        # ---------------Debug--------------
        if options["debug"]:
            print(f"\nTime to precess whole DataSet: {time() - t} seconds")
            print(f"\nNeural network accuracy on trainDataSet: {accuracyTrain[-1]}%")
            print(f"Neural network accuracy on validationDataSet: {accuracyValidation[-1]}%")

        # ---------------Graph--------------
        plt.plot(range(len(accuracyTrain)), accuracyTrain, label="train")
        plt.plot(range(len(accuracyValidation)), accuracyValidation, label="validation")
        plt.title("Neural network\nDataSet:{}, BatchSize:{}, Epoch:{}, InitialRate:{}, Decay:{}".format(len(trainDataSet), hp.batchSize, hp.epoch, hp.initialLearningRate, hp.learnRateDecay))
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(loc="upper left")
        imageName = "hyperParameters/dataset{}_batch{}_epoch{}_lr{}_decay{}.png".format(len(trainDataSet), hp.batchSize, hp.epoch, hp.initialLearningRate, hp.learnRateDecay)

        if options["graph"]:
            plt.show()

        if options["saveCSV"]:
            plt.savefig(imageName)
            path = "/home/elmos/Desktop/ai/digits/"
            with open(path + "hyperParameters/HyperParameters.csv", "a") as file:
                file.write("{},{},{},{},{},{}\n".format(len(trainDataSet), hp.batchSize, hp.epoch, hp.initialLearningRate, hp.learnRateDecay, f"file://{path}{imageName}"))

        # ---------------Save neural--------------
        tmp = input("\nDo you want to save the neural network? y/n ")
        while tmp not in ['y', 'n']:
            tmp = input("Do you want to save the neural network? y/n ")

        if tmp == 'y':
            name = input("Enter a name for the save: ")
            self.ToJson(name)

    def DataPointCost(self, dataPoint):
        outputs = self.CalculateOutputs(dataPoint.input)
        return self.Cost(outputs, dataPoint.target)

    def BatchCost(self, batch):
        cost = 0
        for dataPoint in batch:
            cost += self.DataPointCost(dataPoint)
        return cost / len(batch)

    def DataSetCost(self, dataSet):
        cost = 0
        for batch in dataSet:
            cost += self.BatchCost(batch)
        return cost / len(dataSet)


    def BatchAccuracy(self, batch):
        nbrGood = 0
        for dataPoint in batch:
            if self.Classify(dataPoint.input) == dataPoint.target.index(1):
                nbrGood += 1
        return nbrGood * 100 / len(batch)

    def DataSetAccuracy(self, dataSet):
        averageAccuracy = 0
        for batch in dataSet:
            averageAccuracy += self.BatchAccuracy(batch)
        return averageAccuracy / len(dataSet)

    def Classify(self, inputs):
        outputs = self.CalculateOutputs(inputs)
        return outputs.index(max(outputs))




class Layer:
    def __init__(self, nbrNodesIn, nbrNodesOut, activationFunction=ActivationFunctions.Relu):
        self.nbrNodesIn = nbrNodesIn
        self.nbrNodesOut = nbrNodesOut

        self.weights = activationFunction.value.weightsInitialization(nbrNodesIn, nbrNodesOut)
        self.biases = np.zeros((nbrNodesOut, 1))


        self.gradientWeights = [[0 for _ in range(nbrNodesIn)] for _ in range(nbrNodesOut)]
        self.gradientBiases = np.zeros((nbrNodesOut, 1))

        # Some values are computed during forward pass and stored here for the backpropagation
        self.weightedSum = np.zeros((nbrNodesOut, 1))
        self.outputs = np.zeros((nbrNodesOut, 1))

        self.Activation = activationFunction.value.function
        self.ActivationDerivative = activationFunction.value.derivative

    def ToJson(self):
        jsonObject = {
            "weights": self.weights,
            "biases": self.biases
        }

        return jsonObject

    def SetActivationFunction(self, activationFunction):
        self.Activation = activationFunction.value.function
        self.ActivationDerivative = activationFunction.value.derivative
        self.weights = activationFunction.value.weightsInitialization(self.nbrNodesIn, self.nbrNodesOut)

    def CalculateOutputs(self, inputs):
        # for nodesOut in range(self.nbrNodesOut):
        #     iOutput = self.biases[nodesOut]
        #     for nodesIn in range(self.nbrNodesIn):
        #         iOutput += inputs[nodesIn] * self.weights[nodesOut][nodesIn]
        #
        #     self.weightedSum[nodesOut] = iOutput

        self.weightedSum = np.dot(self.weights, inputs)

        # Run every weightedSum through the activation function
        self.outputs = self.Activation(self.weightedSum)

        return self.outputs

    def UpdateGradient(self, oldLayer, oldNodesValues, previousOutputs):
        newNodeValues = []

        for nodesOut in range(self.nbrNodesOut):
            newNodeValue = 0
            for oldNodesOut in range(oldLayer.nbrNodesOut):
                newNodeValue += oldLayer.weights[oldNodesOut][nodesOut] * oldNodesValues[oldNodesOut]

            newNodeValue *= self.ActivationDerivative(self.weightedSum, nodesOut)
            newNodeValues.append(newNodeValue)

            self.gradientBiases[nodesOut] += newNodeValue
            for nodesIn in range(self.nbrNodesIn):
                self.gradientWeights[nodesOut][nodesIn] += previousOutputs[nodesIn] * newNodeValue

        return newNodeValues

    def ApplyGradient(self, learningRate):
        for nodesOut in range(self.nbrNodesOut):
            for nodesIn in range(self.nbrNodesIn):
                # dividing by batch size to take the average error over all the training data in the batch
                self.weights[nodesOut][nodesIn] -= self.gradientWeights[nodesOut][nodesIn] * learningRate

            self.biases[nodesOut] -= self.gradientBiases[nodesOut] * learningRate

        self.gradientWeights = [[0 for _ in range(self.nbrNodesIn)] for _ in range(self.nbrNodesOut)]
        self.gradientBiases = [0 for _ in range(self.nbrNodesOut)]
