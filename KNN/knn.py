from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt


def kNNClassifyMethod(data, dataSet, labels, k):
    lineNum = dataSet.shape[0]
    diffMat = tile(data, (lineNum, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistances = distances.argsort()
    classCount = {}
    for i in range(k):
        label = labels[sortedDistances[i]]
        classCount[label] = classCount.get(label, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), None, operator.itemgetter(1), True)

    return sortedClassCount[0][0]


def file2Metrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    returnMatrix = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMatrix[index, :] = listFromLine[0:3]
        label = int(listFromLine[-1])
        classLabelVector.append(label)
        index += 1

    print returnMatrix[:, 1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(returnMatrix[:, 1], returnMatrix[:, 2], 15 * array(classLabelVector), 15 * array(classLabelVector))
    plt.show()
    return returnMatrix, classLabelVector


matrix, labels = file2Metrix("./datingTestSet.txt")
label = kNNClassifyMethod((15669, 4.680098, 0.191283), matrix, labels, 10)
print label
