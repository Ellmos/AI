from neural import NeuralNetwork, HyperParameters
from ActivationFunctions import ActivationFunctions
from CostFunctions import CostFunctions
from dataLoader import LoadDataset

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
    t = time()
    (x_train, y_train), (x_test, y_test) = LoadDataset(batchSize)

    dataSet = []
    for (batch_x, batch_y) in zip(x_train, y_train):
        newBatch = []
        for i in range(len(batch_y)):
            output = [0 for _ in range(10)]
            y = batch_y[i]
            output[y-1] = 1
            newBatch.append(Data(batch_x[i], output))
        dataSet.append(newBatch)

    testDataSet = []
    for i in range(len(x_test)):
        output = [0 for _ in range(10)]
        y = y_test[i]
        output[y] = 1
        testDataSet.append(Data(x_test[i], output))

    # i = [3, 5, 7, 9, 0, 13, 15, 17, 4, 1]
    # trainDataSet = [trainDataSet[j] for j in i]
    shuffle(dataSet)
    shuffle(testDataSet)

    split = int(len(dataSet) * 0.90)
    trainDataSet = dataSet[:split]
    validationDataSet = dataSet[split:]

    print("DatSet created in:", time() - t, "seconds")
    return trainDataSet, validationDataSet, testDataSet


# Create neural network
hp = HyperParameters()
neural = NeuralNetwork([784, 30, 10])
neural.SetActivationFunctions(hp.activationFunction, hp.outputActivationType)
neural.SetCostFunction(hp.costFunction)


# Create dataSet
trainDataSet, validationDataSet, testDataSet = GenerateDataSet(hp.batchSize)
trainDataSet = trainDataSet[:100 // hp.batchSize]
validationDataSet = validationDataSet[:100 // hp.batchSize]
testDataSet = testDataSet[:1000 // hp.batchSize]


nbrBatch = len(trainDataSet)
printBatch = nbrBatch // 10 if nbrBatch // 10 != 0 else 1

t = time()
costs = []
print("\n------------------Learning-----------------------", end="")
for currentEpoch in range(hp.epoch):
    print("\n--Epoch {} out of {}--".format(currentEpoch+1, hp.epoch))

    learningRate = hp.initialLearningRate * (1 / (1 + hp.learnRateDecay * currentEpoch))
    shuffle(trainDataSet)
    for i in range(nbrBatch):
        if i % printBatch == 0:
            print("Batch {} out of {}".format(i, nbrBatch))

        batch = trainDataSet[i]

        neural.Learn(batch, learningRate)
    costs.append(neural.DataSetCost(validationDataSet))

print(time() - t)

plt.plot(range(len(costs)), costs)
plt.title("Neural network\nDataSet:{}, BatchSize:{}, Epoch:{}, InitialRate:{}, Decay:{}".format(len(trainDataSet), hp.batchSize, hp.epoch, hp.initialLearningRate, hp.learnRateDecay))
plt.xlabel("Epoch")
plt.ylabel("Cost")

path = "/home/elmos/Desktop/ai/digits/"
imageName = "hyperParameters/dataset{}_batch{}_epoch{}_lr{}_decay{}.png".format(len(trainDataSet), hp.batchSize, hp.epoch, hp.initialLearningRate, hp.learnRateDecay)
plt.savefig(imageName)

with open(path + "hyperParameters/HyperParameters.csv", "a") as file:
    file.write("{},{},{},{},{},{}\n".format(len(trainDataSet), hp.batchSize, hp.epoch, hp.initialLearningRate, hp.learnRateDecay, "file://" + path + imageName))



print("\nNeural network cost:", costs[-1])
plt.show()


tmp = input("Do you want to save the neural network? y/n ")
while tmp not in ['y', 'n']:
    tmp = input("Do you want to save the neural network? y/n ")

if tmp == 'y':
    neural.ToJson("neuralSave")

