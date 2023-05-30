from neural import *
from dataLoader import LoadDataset

from random import shuffle
from dataclasses import dataclass
from time import time


@dataclass
class Data:
    input: list
    target: list



def GenerateDataSet(batchSize=0):
    print("--------------Creating Dataset-------------------")
    t = time()
    (x_train, y_train), (x_validation, y_validation), (x_test, y_test) = LoadDataset(batchSize)


    trainDataSet = []
    for (batch_x, batch_y) in zip(x_train, y_train):
        newBatch = []
        for i in range(len(batch_y)):
            output = [0 for _ in range(10)]
            y = batch_y[i]
            output[y] = 1
            newBatch.append(Data(batch_x[i], output))
        trainDataSet.append(newBatch)

    validationDataSet = []
    for i in range(len(x_validation)):
        output = [0 for _ in range(10)]
        y = y_validation[i]
        output[y] = 1
        validationDataSet.append(Data(x_validation[i], output))

    testDataSet = []
    for i in range(len(x_test)):
        output = [0 for _ in range(10)]
        y = y_test[i]
        output[y] = 1
        testDataSet.append(Data(x_test[i], output))

    shuffle(trainDataSet)


    print("DatSet created in:", time() - t, "seconds")
    return trainDataSet, validationDataSet, testDataSet


# Create neural network
hp = HyperParameters()
neural = NeuralNetwork([784, 30, 10], hp)

# Create dataSet
trainDataSet, validationDataSet, testDataSet = GenerateDataSet(hp.batchSize)

# trainDataSet = trainDataSet[:1000 // hp.batchSize]&
# validationDataSet = validationDataSet[:1000]
# testDataSet = testDataSet[:1000]

# Learning
options = {"debug": True, "graph": True, "saveCSV": True}
neural.Learn(trainDataSet, validationDataSet, hp, options)

