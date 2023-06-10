import numpy as np
import cv2
from dataLoader import ReadMnistFiles
from random import randint


def RotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


def TranslateImage(image, x, y):
    M = np.float32([[1, 0, x],
                    [0, 1, y]])
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

def ShowImage(image):
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def LoadNormalMnist():
    directory = './data/mnist/'
    training_images_path = directory + 'train-images.idx3-ubyte'
    training_labels_path = directory + 'train-labels.idx1-ubyte'
    test_images_path = directory + 't10k-images.idx3-ubyte'
    test_labels_path = directory + 't10k-labels.idx1-ubyte'

    mnistTrainImages, mnistTrainLabels = ReadMnistFiles(training_images_path, training_labels_path)
    mnistTestImages, mnistTestLabels = ReadMnistFiles(test_images_path, test_labels_path)
    return (mnistTrainImages, mnistTrainLabels), (mnistTestImages, mnistTestLabels)



def ModifyDataSet(images, labels, trainDataSet):
    labels = labels.tolist()
    dir = "./data/modifiedMnist/"
    if trainDataSet:
        imageFile = open(dir + "modifiedTrainImages.bytes", "wb")
        labelFile = open(dir + "modifiedTrainLabels.bytes", "wb")
    else:
        imageFile = open(dir + "modifiedTestImages.bytes", "wb")
        labelFile = open(dir + "modifiedTestLabels.bytes", "wb")

    length = len(images)
    imageFile.write(length.to_bytes(8, "little"))
    labelFile.write(length.to_bytes(8, "little"))

    rotationRange = 10
    translationRange = 4
    for i in range(len(images)):
        angle = randint(-rotationRange, rotationRange)
        xTranslation, yTranslation = randint(-translationRange, translationRange), randint(-translationRange, translationRange)

        image = images[i].reshape((28, 28))


        rotated = RotateImage(image, angle)
        translated = TranslateImage(rotated, xTranslation, yTranslation)

        noise = np.zeros(image.shape, np.float64)
        cv2.randn(noise, 0, 0.1)
        noise = np.maximum(noise, 0)
        noised = cv2.add(translated, noise).reshape(784)

        img = noised * (1.0 / noised.max())

        imageFile.write(img.tobytes())
        labelFile.write(labels[i].to_bytes(1, "little"))

        if i % 200 == 0:
            print(i)

    imageFile.close()
    labelFile.close()



image = cv2.imread("resized.png", cv2.IMREAD_GRAYSCALE)


(mnistTrainImages, mnistTrainLabels), (mnistTestImages, mnistTestLabels) = LoadNormalMnist()
ModifyDataSet(mnistTrainImages, mnistTrainLabels, True)
