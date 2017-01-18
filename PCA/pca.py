#coding=utf-8
from numpy import *


def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float, line) for line in stringArr]
    return mat(datArr)


def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat)
    meanRemoved = dataMat - meanVals  #去均值
    covMat = cov(meanRemoved)      #求协方差矩阵
    eigVals, eigVects = linalg.eig(mat(covMat))
    print eigVals
    print eigVects
    eigValInd = argsort(eigVals)  #对特征向量进行排序
    eigValInd = eigValInd[:-(topNfeat + 1):-1]  #求出前N个特征值
    redEigVects = eigVects[:, eigValInd]  #求出特征值对应的特征向量
    lowDDataMat = meanRemoved * redEigVects  # transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat


def replaceNanWithMean():
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:, i].A))[0], i])  # values that are not NaN (a number)
        datMat[nonzero(isnan(datMat[:, i].A))[0], i] = meanVal  # set NaN values to mean
    return datMat


mat2 = loadDataSet("trainSet.txt")
pca(mat2)


