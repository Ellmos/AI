import numpy as np
import cv2
from random import randint, random

from dataLoader import ReadDataSetFiles
import os

def RotateImage(image, angle):
    imageCenter = tuple(np.array(image.shape[1::-1]) / 2)
    rotMat = cv2.getRotationMatrix2D(imageCenter, angle, 1.0)
    rotated = cv2.warpAffine(image, rotMat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return rotated


def TranslateImage(image, x, y):
    M = np.float32([[1, 0, x],
                    [0, 1, y]])
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

def ShowImage(image):
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def ModifyMnist(images, labels, trainDataSet):
    labels = labels.tolist()
    directory = os.path.dirname(os.path.abspath(__file__)) + '/modifiedMnist/'
    if trainDataSet:
        imageFile = open(directory + "modifiedTrainImages.bytes", "wb")
        labelFile = open(directory + "modifiedTrainLabels.bytes", "wb")
    else:
        imageFile = open(directory + "modifiedTestImages.bytes", "wb")
        labelFile = open(directory + "modifiedTestLabels.bytes", "wb")

    length = len(images)
    imageFile.write(length.to_bytes(8, "little"))
    labelFile.write(length.to_bytes(8, "little"))

    rotationRange = 10
    translationRange = 4
    for i in range(length):
        angle = randint(-rotationRange, rotationRange)
        xTranslation, yTranslation = randint(-translationRange, translationRange), randint(-translationRange, translationRange)

        image = images[i].reshape((28, 28))


        rotated = RotateImage(image, angle)
        translated = TranslateImage(rotated, xTranslation, yTranslation)

        noise = np.zeros(image.shape, np.float64)
        cv2.randn(noise, 0, random()/10)
        noise = np.maximum(noise, 0)
        noised = cv2.add(translated, noise)

        noised = np.minimum(noised, 1).reshape(784)

        imageFile.write(noised.tobytes())
        labelFile.write(labels[i].to_bytes(1, "little"))

        if i % 200 == 0:
            print(i)

    imageFile.close()
    labelFile.close()




(mnistTrainImages, mnistTrainLabels), (mnistTestImages, mnistTestLabels) = ReadDataSetFiles(mnist=100, modifiedMnist=0, own=0)
ModifyMnist(mnistTrainImages, mnistTrainLabels, True)
ModifyMnist(mnistTestImages, mnistTestLabels, False)
