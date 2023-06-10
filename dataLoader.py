import struct
from array import array
import numpy as np
from time import time
from dataclasses import dataclass
from random import shuffle


def ReadMnistFiles(imagesPath, labelsPath):
    labels = []
    # Read labels file
    with open(labelsPath, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        labels = array("B", file.read())

    # Read images file
    with open(imagesPath, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
        image_data = array("B", file.read())

    images = [[]] * size
    for i in range(size):
        img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
        img = img * (1.0 / img.max())
        images[i] = img

    return np.array(images), np.array(labels)


def ReadOwnFiles(imagesPath, labelsPath):
    images = []
    imageFile = open(imagesPath, "rb")
    labelFile = open(labelsPath, "rb")
    iteration = int.from_bytes(imageFile.read(8), "little")
    labelFile.read(8)  # the iteration value is also written at the first 8 bytes of the labelFile and can be skipped

    for i in range(iteration):
        imgBytes = imageFile.read(8 * 784)
        images.append(np.frombuffer(imgBytes))

    labels = [int.from_bytes(labelFile.read(1), "little") for _ in range(iteration)]

    return np.array(images), np.array(labels)


def ReadDataSetFiles():
    # --------------Load Mnist DataSet---------------------
    directory = './data/mnist/'
    training_images_path = directory + 'train-images.idx3-ubyte'
    training_labels_path = directory + 'train-labels.idx1-ubyte'
    test_images_path = directory + 't10k-images.idx3-ubyte'
    test_labels_path = directory + 't10k-labels.idx1-ubyte'

    mnistTrainImages, mnistTrainLabels = ReadMnistFiles(training_images_path, training_labels_path)
    mnistTestImages, mnistTestLabels = ReadMnistFiles(test_images_path, test_labels_path)

    # --------------Load Own DataSet---------------------
    directory = './data/ownDataSet/'
    training_images_path = directory + 'ownTrainImages.bytes'
    training_labels_path = directory + 'ownTrainLabels.bytes'
    test_images_path = directory + 'ownTestImages.bytes'
    test_labels_path = directory + 'ownTestLabels.bytes'
    ownTrainImages, ownTrainLabels = ReadOwnFiles(training_images_path, training_labels_path)
    ownTestImages, ownTestLabels = ReadOwnFiles(test_images_path, test_labels_path)

    # --------------Join DataSets---------------------
    trainImages = np.concatenate((mnistTrainImages, ownTrainImages))
    trainLabels = np.concatenate((mnistTrainLabels, ownTrainLabels))

    testImages = np.concatenate((mnistTestImages, ownTestImages))
    testLabels = np.concatenate((mnistTestLabels, ownTestLabels))

    return (trainImages, trainLabels), (testImages, testLabels)


@dataclass
class Data:
    input: list
    target: list


def GenerateDataSet(batchSize):
    print("--------------Creating Dataset-------------------")
    t = time()
    (trainImages, trainLabels), (testImages, testLabels) = ReadDataSetFiles()

    if batchSize == 0:
        nbrBatch = 1
        batchSize = len(trainImages)
    else:
        nbrBatch = len(trainImages) // batchSize

    trainDataSet = []
    for i in range(0, nbrBatch):
        newBatch = []
        for j in range(batchSize):
            image = trainImages[i * batchSize + j]
            label = trainLabels[i * batchSize + j]
            output = [0 for _ in range(10)]
            output[label] = 1
            newBatch.append(Data(image, output))
        trainDataSet.append(newBatch)

    testDataSet = []
    for i in range(len(testImages)):
        output = [0 for _ in range(10)]
        y = testLabels[i]
        output[y] = 1
        testDataSet.append(Data(testImages[i], output))

    shuffle(trainDataSet)

    print("DatSet created in", time() - t, "seconds")
    return trainDataSet, [testDataSet]
