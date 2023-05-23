from ActivationFunctions import ActivationFunctions
from CostFunctions import CostFunctions

def CostDerivative(outputValue, targetValue):
    return outputValue - targetValue


class NeuralNetwork:
    def __init__(self, layersSizes, CostFunction=CostFunctions.CrossEntropy):
        self.nbrLayers = len(layersSizes) - 1
        self.layers = [Layer(layersSizes[i], layersSizes[i + 1]) for i in range(self.nbrLayers)]
        self.Cost = CostFunction.value.function
        self.CostDerivative = CostFunction.value.derivative

    def SetActivationFunction(self, ActivationFunction):
        for layer in self.layers:
            layer.Activation = ActivationFunction.value.function
            layer.ActivationDerivative = ActivationFunction.value.derivative

    def SetOutputActivationFunction(self, ActivationFunction):
        self.layers[-1].SetActivationFunction(ActivationFunction)

    def SetCostFunction(self, CostFunction):
        self.Cost = CostFunction.value.function
        self.CostDerivative = CostFunction.value.derivative

    def CalculateOutputs(self, inputs):
        for layer in self.layers:
            inputs = layer.CalculateOutputs(inputs)
        return inputs

    def Classify(self, inputs):
        outputs = self.CalculateOutputs(inputs)
        return outputs.index(max(outputs))


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


    def Learn(self, dataPoints, learningRate):
        for dataPoint in dataPoints:
            self.CalculateOutputs(dataPoint.input)


            outputLayer = self.layers[-1]
            previousOutputs = self.layers[-2].outputs if self.nbrLayers >= 2 else dataPoint.input

            # Compute the nodes values of the output layer and update its gradients
            nodesValues = []
            for nodesOut in range(outputLayer.nbrNodesOut):
                activationDerivative = outputLayer.ActivationDerivative(outputLayer.weightedSum, nodesOut)
                costDerivative = CostDerivative(outputLayer.outputs[nodesOut], dataPoint.target[nodesOut])
                currentNodeValue = costDerivative * activationDerivative
                nodesValues.append(currentNodeValue)

                outputLayer.gradientBiases[nodesOut] += currentNodeValue
                for nodesIn in range(outputLayer.nbrNodesIn):
                    outputLayer.gradientWeights[nodesOut][nodesIn] += previousOutputs[nodesIn] * currentNodeValue


            # outputLayer.UpdateGradient(nodesValues, previousOutputs)


            # Go back through the layers, compute the corresponding node values and update the gradient at the same time
            for i in range(2, self.nbrLayers + 1):
                previousOutputs = self.layers[-i - 1].outputs if i < self.nbrLayers else dataPoint.input

                currentLayer = self.layers[-i]
                nodesValues = currentLayer.CalculateHiddenLayerNodesValues(self.layers[-i + 1], nodesValues, previousOutputs)
                # layer.UpdateGradient(nodesValues, previousOutputs)

        for layer in self.layers:
            layer.ApplyGradient(learningRate / len(dataPoints))


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

    # def CalculateOutputLayerNodesValues(self, targets):
    #     nodesValues = []
    #     for nodesOut in range(self.nbrNodesOut):
    #         nodesValues.append(CostDerivative(self.outputs[nodesOut], targets[nodesOut]) * self.ActivationDerivative(self.weightedSum, nodesOut))
    #
    #     return nodesValues

    def CalculateHiddenLayerNodesValues(self, oldLayer, oldNodesValues, previousOutputs):
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

    # def UpdateGradient(self, nodesValues, previousOutputs):
    #     for nodesOut in range(self.nbrNodesOut):
    #         for nodesIn in range(self.nbrNodesIn):
    #             self.gradientWeights[nodesOut][nodesIn] += previousOutputs[nodesIn] * nodesValues[nodesOut]
    #
    #         self.gradientBiases[nodesOut] += nodesValues[nodesOut]

    def ApplyGradient(self, learningRate):
        for nodesOut in range(self.nbrNodesOut):
            for nodesIn in range(self.nbrNodesIn):
                # dividing by batch size to take the average error over all the training data in the batch
                self.weights[nodesOut][nodesIn] -= self.gradientWeights[nodesOut][nodesIn] * learningRate

            self.biases[nodesOut] -= self.gradientBiases[nodesOut] * learningRate

        self.gradientWeights = [[0 for _ in range(self.nbrNodesIn)] for _ in range(self.nbrNodesOut)]
        self.gradientBiases = [0 for _ in range(self.nbrNodesOut)]
