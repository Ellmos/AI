from neural import NeuralNetwork
from dataclasses import dataclass
import matplotlib.pyplot as plt
from dataLoader import LoadMnistDataset
from ActivationFunctions import ActivationFunctions
from CostFunctions import CostFunctions
from time import time


@dataclass
class Data:
    input: list
    target: list



def GenerateDataSet(batchSize):
    print("--------------Creating Dataset-------------------")
    (x_train, y_train), (x_test, y_test) = LoadMnistDataset(batchSize)
    trainDataSet = []
    for (batch_x, batch_y) in zip(x_train, y_train):
        newBatch = []
        for i in range(len(batch_y)):
            output = [0 for _ in range(9)]
            y = batch_y[i]
            if y != 0:
                output[y-1] = 1
            newBatch.append(Data(batch_x[i], output))
        trainDataSet.append(newBatch)


    testDataSet = []
    for i in range(len(x_test)):
        output = [0 for _ in range(9)]
        y = y_test[i]
        if y != 0:
            output[y-1] = 1
        testDataSet.append(Data(x_test[i], output))


    return trainDataSet, testDataSet



batchSize = 36
initialLearningRate = 0.5
learnRateDecay = 0.075
epoch = 10

t = time()
trainDataSet, testDataSet = GenerateDataSet(batchSize)
print("DatSet created in:", time() - t, "seconds")
trainDataSet = trainDataSet[:10]
testDataSet = testDataSet[:100]

nbrBatch = len(trainDataSet)
printBatch = nbrBatch // 10

neural = NeuralNetwork([784, 16, 9], CostFunctions.MeanSquare)
neural.SetActivationFunction(ActivationFunctions.LeakyRelu)
neural.SetOutputActivationFunction(ActivationFunctions.Softmax)


costs = []
print("\n------------------Learning-----------------------", end="")
for currentEpoch in range(epoch):
    print("\n--Epoch {} out of {}--".format(currentEpoch+1, epoch))
    learningRate = initialLearningRate * (1 / (1 + learnRateDecay * currentEpoch))
    for i in range(nbrBatch):
        batch = trainDataSet[i]

        neural.Learn(batch, learningRate)
        costs.append(neural.AllDataPointsCost(trainDataSet[0]))
        if i % printBatch == 0:
            print("Batch {} out of {}".format(i, nbrBatch))

plt.plot(range(epoch*nbrBatch), costs)
plt.show()

print("\nNeural network cost:", costs[-1])
