from neural import NeuralNetwork
from ActivationFunctions import ActivationFunctions
from CostFunctions import CostFunctions
from dataLoader import LoadMnistDataset

from random import shuffle
from dataclasses import dataclass
import matplotlib.pyplot as plt
from time import time


@dataclass
class Data:
    input: list
    target: list



def GenerateDataSet(batchSize=0):
    print("--------------Creating Dataset-------------------")
    (x_train, y_train), (x_test, y_test) = LoadMnistDataset(batchSize)
    trainDataSet = []
    for (batch_x, batch_y) in zip(x_train, y_train):
        newBatch = []
        for i in range(len(batch_y)):
            output = [0 for _ in range(10)]
            y = batch_y[i]
            output[y-1] = 1
            newBatch.append(Data(batch_x[i], output))
        trainDataSet.append(newBatch)

    testDataSet = []
    for i in range(len(x_test)):
        output = [0 for _ in range(10)]
        y = y_test[i]
        output[y] = 1
        testDataSet.append(Data(x_test[i], output))

    i = [3, 5, 7, 9, 0, 13, 15, 17, 4, 1]
    trainDataSet = [trainDataSet[j] for j in i]
    shuffle(trainDataSet)
    shuffle(testDataSet)

    return trainDataSet, testDataSet



batchSize = 1
initialLearningRate = 0.5
learnRateDecay = 0.005
epoch = 500

t = time()
trainDataSet, testDataSet = GenerateDataSet(batchSize)

print("DatSet created in:", time() - t, "seconds")
# trainDataSet = trainDataSet[:2]
# testDataSet = testDataSet[:1000]


nbrBatch = len(trainDataSet)
printBatch = nbrBatch // 10 if nbrBatch // 10 != 0 else 1

neural = NeuralNetwork([784, 30, 10], CostFunctions.CrossEntropy)
neural.SetActivationFunction(ActivationFunctions.LeakyRelu)
neural.SetOutputActivationFunction(ActivationFunctions.Softmax)


costs = []
print("\n------------------Learning-----------------------", end="")
for currentEpoch in range(epoch):
    learningRate = initialLearningRate * (1 / (1 + learnRateDecay * currentEpoch))
    if currentEpoch % 100 == 0:
        print("\n--Epoch {} out of {}--".format(currentEpoch+1, epoch))
    shuffle(trainDataSet)
    for i in range(nbrBatch):
        if i % printBatch == 0:
            pass # print("Batch {} out of {}".format(i, nbrBatch))

        batch = trainDataSet[i]

        neural.Learn(batch, learningRate)
    costs.append(neural.DataSetCost(trainDataSet))


plt.plot(range(len(costs)), costs)
plt.title("Neural network\nDataSet:{}, BatchSize:{}, Epoch:{}, InitialRate:{}, Decay:{}".format(len(trainDataSet), batchSize, epoch, initialLearningRate, learnRateDecay))
plt.xlabel("Epoch")
plt.ylabel("Cost")
# plt.savefig("dataset{}_batch{}_epoch{}_lr{}_decay{}.png".format(len(trainDataSet), batchSize, epoch, initialLearningRate, learnRateDecay))
plt.show()

print("\nNeural network cost:", costs[-1])

for batch in trainDataSet:
    for data in batch:
        print(neural.Classify(data.input), data.target.index(max(data.target)), neural.CalculateOutputs(data.input))