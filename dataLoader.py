import numpy as np
import struct
from array import array
import os


# explicit function to normalize array
def Normalize(arr):
    norm_arr = []
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = ((i - min(arr)) / diff_arr) + 1
        norm_arr.append(temp)
    return norm_arr


def ReadMnistFiles(imagesFilepath, labelsFilepath, batchSize):
    labels = []
    # Read labels file
    with open(labelsFilepath, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        labels = array("B", file.read())


    # Read images file
    with open(imagesFilepath, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
        image_data = array("B", file.read())

    images = [[]] * size
    for i in range(size):
        img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
        img = img * (1.0/img.max())
        images[i] = img

    if batchSize == 0:
        return images, labels

    split = int(len(labels) * 0.83)
    (x_validation, y_validation) = (images[split:], labels[split:])


    batchNumber = len(images) / batchSize
    images = np.array_split(np.array(images), batchNumber)
    labels = np.array_split(np.array(labels), batchNumber)
    return (images, labels), (x_validation, y_validation)


# def LoadPersonalDataset():
#     dir = "./data/number-generator/output/"
#     files = os.listdir(dir)
#     with open(dir+files[0], "rb") as image:
#         print(struct.unpack(">IIII", image.read(16)))
#         f = image.read(8)
#         print(f)


def LoadDataset(batchSize):
    # LoadPersonalDataset()
    dataDirectory = './data/'
    training_images_filepath = dataDirectory + 'train-images.idx3-ubyte'
    training_labels_filepath = dataDirectory + 'train-labels.idx1-ubyte'
    test_images_filepath = dataDirectory + 't10k-images.idx3-ubyte'
    test_labels_filepath = dataDirectory + 't10k-labels.idx1-ubyte'

    (x_train, y_train), (x_validation, y_validation) = ReadMnistFiles(training_images_filepath, training_labels_filepath, batchSize)
    x_test, y_test = ReadMnistFiles(test_images_filepath, test_labels_filepath, 0)

    if batchSize == 0:
        return ([x_train], [y_train]), (x_test, y_test)

    return (x_train, y_train), (x_validation, y_validation), (x_test, y_test)

