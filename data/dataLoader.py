import struct
from array import array
import numpy as np
from time import time
from dataclasses import dataclass
from random import shuffle
import os



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


def ReadDataSetFiles(mnist, modifiedMnist, own):
    trainingDataSets = []
    testDatasets = []
    nbrDataset = 3 - (mnist, modifiedMnist, own).count(0)
    if nbrDataset == 0:
        raise Exception("You need at least one DataSet")


    # --------------Load Mnist DataSet---------------------
    if mnist:

        directory = os.path.dirname(os.path.abspath(__file__)) + '/mnist/'
        training_images_path = directory + 'train-images.idx3-ubyte'
        training_labels_path = directory + 'train-labels.idx1-ubyte'
        test_images_path = directory + 't10k-images.idx3-ubyte'
        test_labels_path = directory + 't10k-labels.idx1-ubyte'

        trainImages, trainLabels = ReadMnistFiles(training_images_path, training_labels_path)
        testImages, testLabels = ReadMnistFiles(test_images_path, test_labels_path)

        lengthTraining = len(trainImages * mnist / 100)
        lengthTest = len(testImages * mnist / 100)
        trainingDataSets.append((trainImages[:lengthTraining], trainLabels[:lengthTraining]))
        testDatasets.append((testImages[:lengthTest], testLabels[:lengthTest]))


    # --------------Load ModifiedMnist DataSet---------------------
    if modifiedMnist:
        directory = os.path.dirname(os.path.abspath(__file__)) + '/modifiedMnist/'
        training_images_path = directory + 'modifiedTrainImages.bytes'
        training_labels_path = directory + 'modifiedTrainLabels.bytes'
        test_images_path = directory + 'modifiedTestImages.bytes'
        test_labels_path = directory + 'modifiedTestLabels.bytes'

        trainImages, trainLabels = ReadOwnFiles(training_images_path, training_labels_path)
        testImages, testLabels = ReadOwnFiles(test_images_path, test_labels_path)

        lengthTraining = len(trainImages * modifiedMnist / 100)
        lengthTest = len(testImages * modifiedMnist / 100)
        trainingDataSets.append((trainImages[:lengthTraining], trainLabels[:lengthTraining]))
        testDatasets.append((testImages[:lengthTest], testLabels[:lengthTest]))


    # --------------Load Own DataSet---------------------
    if own:
        directory = os.path.dirname(os.path.abspath(__file__)) + '/ownDataSet/'
        training_images_path = directory + 'ownTrainImages.bytes'
        training_labels_path = directory + 'ownTrainLabels.bytes'
        test_images_path = directory + 'ownTestImages.bytes'
        test_labels_path = directory + 'ownTestLabels.bytes'

        trainImages, trainLabels = ReadOwnFiles(training_images_path, training_labels_path)
        testImages, testLabels = ReadOwnFiles(test_images_path, test_labels_path)

        lengthTraining = len(trainImages * own / 100)
        lengthTest = len(testImages * own / 100)
        trainingDataSets.append((trainImages[:lengthTraining], trainLabels[:lengthTraining]))
        testDatasets.append((testImages[:lengthTest], testLabels[:lengthTest]))


    # --------------Merge DataSets---------------------
    trainImages, trainLabels = trainingDataSets[0]
    for i in range(1, nbrDataset):
        images, labels = trainingDataSets[1]
        trainImages = np.concatenate((trainImages, images))
        trainLabels = np.concatenate((trainLabels, labels))

    testImages, testLabels = testDatasets[0]
    for i in range(1, nbrDataset):
        images, labels = testDatasets[1]
        testImages = np.concatenate((testImages, images))
        testLabels = np.concatenate((testLabels, labels))


    return (trainImages, trainLabels), (testImages, testLabels)


@dataclass
class Data:
    input: list
    target: list


def LoadDataSets(batchSize, mnist=100, modifiedMnist=100, own=100):
    print("--------------Creating Dataset-------------------")
    t = time()
    (trainImages, trainLabels), (testImages, testLabels) = ReadDataSetFiles(mnist, modifiedMnist, own)

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
