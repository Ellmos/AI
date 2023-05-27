from ActivationFunctions import ActivationFunctions
from CostFunctions import CostFunctions
import json

import matplotlib.pyplot as plt
from random import shuffle
from time import time



class HyperParameters:
    def __init__(self):
        self.activationFunction = ActivationFunctions.Relu
        self.outputActivationType = ActivationFunctions.Softmax
        self.costFunction = CostFunctions.CrossEntropy
        self.initialLearningRate = 0.025
        self.learnRateDecay = 0.075
        self.batchSize = 32
        self.epoch = 10


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

    def FeedBatch(self, dataPoints, learningRate):
        for dataPoint in dataPoints:
            self.CalculateOutputs(dataPoint.input)


            outputLayer = self.layers[-1]
            previousOutputs = self.layers[-2].outputs if self.nbrLayers >= 2 else dataPoint.input

            # Compute the nodes values of the output layer and update its gradients
            nodesValues = []
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
            layer.ApplyGradient(learningRate / len(dataPoints))


    def Learn(self, trainDataSet, validationDataSet, hp, options):

        nbrBatch = len(trainDataSet)
        printBatch = nbrBatch // 10 if nbrBatch // 10 != 0 else 1

        accuracy = []

        t = time()
        print("\n------------------Learning-----------------------", end="")
        for currentEpoch in range(hp.epoch):
            if options["debug"]:
                print("\n--Epoch {} out of {}--".format(currentEpoch + 1, hp.epoch))

            learningRate = hp.initialLearningRate * (1 / (1 + hp.learnRateDecay * currentEpoch))
            shuffle(trainDataSet)
            for i in range(nbrBatch):
                if options["debug"] and i % printBatch == 0:
                    print("Batch {} out of {}".format(i, nbrBatch))

                batch = trainDataSet[i]

                self.FeedBatch(batch, learningRate)
            accuracy.append(self.DataSetAccuracy(validationDataSet))


        # ---------------Debug--------------
        if options["debug"]:
            print(time() - t)
            print(f"\nNeural network accuracy: {accuracy[-1]}%")

        # ---------------Graph--------------
        plt.plot(range(len(accuracy)), accuracy, label="accuracy")
        plt.title("Neural network\nDataSet:{}, BatchSize:{}, Epoch:{}, InitialRate:{}, Decay:{}".format(len(trainDataSet), hp.batchSize, hp.epoch, hp.initialLearningRate, hp.learnRateDecay))
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        imageName = "hyperParameters/dataset{}_batch{}_epoch{}_lr{}_decay{}.png".format(len(trainDataSet), hp.batchSize, hp.epoch, hp.initialLearningRate, hp.learnRateDecay)

        if options["graph"]:
            plt.show()

        if options["saveCSV"]:
            plt.savefig(imageName)
            path = "/home/elmos/Desktop/ai/digits/"
            with open(path + "hyperParameters/HyperParameters.csv", "a") as file:
                file.write("{},{},{},{},{},{}\n".format(len(trainDataSet), hp.batchSize, hp.epoch, hp.initialLearningRate, hp.learnRateDecay, f"file://{path}{imageName}"))

        # ---------------Save neural--------------
        tmp = input("Do you want to save the neural network? y/n ")
        while tmp not in ['y', 'n']:
            tmp = input("Do you want to save the neural network? y/n ")

        if tmp == 'y':
            name = input("Enter a name for the save: ")
            self.ToJson(name)

    def DataPointCost(self, dataPoint):
        outputs = self.CalculateOutputs(dataPoint.input)
        return self.Cost(outputs, dataPoint.target)

    def BatchCost(self, dataPoints):
        cost = 0
        for data in dataPoints:
            cost += self.DataPointCost(data)

        return cost / len(dataPoints)

    def DataSetCost(self, dataSet):
        cost = 0
        for batch in dataSet:
            cost += self.BatchCost(batch)

        return cost / len(dataSet)


    def DataSetAccuracy(self, dataSet):
        good = 0
        for data in dataSet:
            if self.Classify(data.input) == data.target.index(1):
                good += 1
        return good * 100 / len(dataSet)

    def Classify(self, inputs):
        outputs = self.CalculateOutputs(inputs)
        return outputs.index(max(outputs))




class Layer:
    def __init__(self, nbrNodesIn, nbrNodesOut, activationFunction=ActivationFunctions.Relu):
        self.nbrNodesIn = nbrNodesIn
        self.nbrNodesOut = nbrNodesOut

        self.weights = activationFunction.value.weightsInitialization(nbrNodesIn, nbrNodesOut)
        self.biases = [0 for _ in range(nbrNodesOut)]

        self.gradientWeights = [[0 for _ in range(nbrNodesIn)] for _ in range(nbrNodesOut)]
        self.gradientBiases = [0 for _ in range(nbrNodesOut)]

        # Some values are computed during forward pass and stored here for the backpropagation
        self.weightedSum = []
        self.outputs = []

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
        self.weightedSum = []
        self.outputs = []
        for nodesOut in range(self.nbrNodesOut):
            iOutput = self.biases[nodesOut]
            for nodesIn in range(self.nbrNodesIn):
                iOutput += inputs[nodesIn] * self.weights[nodesOut][nodesIn]

            self.weightedSum.append(iOutput)

        # Run every weightedSum through the activation function
        for i in range(self.nbrNodesOut):
            self.outputs.append(self.Activation(self.weightedSum, i))

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
