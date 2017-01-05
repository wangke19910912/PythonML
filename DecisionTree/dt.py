from math import log


# 计算说香农熵
def calcShannonEnt(dataSet):
    dataLen = len(dataSet)
    labelCounts = getLabelCounts(dataSet)
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / dataLen
        shannonEnt -= prob * log(prob, 2)

    return shannonEnt


def chooseBestFeatureToSplit(dataSet):
    dataLen = len(dataSet)
    dataSetShannon = calcShannonEnt(dataSet)
    numFeatures = len(dataSet[0].split('\t')) - 1
    bestInfoGain = 0.0
    bestFeature = 1
    for i in range(numFeatures):
        uniqueVals = getFeatureCategory(dataSet, i)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = getSplitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = dataSetShannon - newEntropy

        print "i:" + float(i)
        print "infoGain:" + float(infoGain)
        if (bestInfoGain > infoGain):
            bestInfoGain = infoGain
            bestFeature = i


def createTree(dataSet, label):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]


def getLabelCounts(dataSet):
    labelCounts = {}
    for data in dataSet:
        label = data.split("\t")[-1]
        if label not in labelCounts:
            labelCounts[label] = 1
        labelCounts[label] += 1
    return labelCounts


def getSplitDataSet(dataSet, featureIndex, value):
    splitData = []
    for data in dataSet:
        formatData = data.split("\t")
        if splitData[featureIndex] == value:
            splitData.append(data)
    return splitData


def getFeatureCategory(dataSet, featureIndex):
    featureCategory = []
    for data in dataSet:
        feature = data.split("\t")[featureIndex]
        if feature not in featureCategory:
            featureCategory.append(feature)
    return featureCategory
