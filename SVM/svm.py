from numpy import *
from time import sleep

def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m):
    j=i #we want to select any J not equal to i
    while (j==i):
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

dataArr,labelArr = loadDataSet('testSet.txt')


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    #数组转化为矩阵
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    b = 0
    #计算矩阵维度R=m*n
    m,n = shape(dataMatrix)
    #初始化一个空白矩阵R=1*m,假设所有点都可以为计算f(x)做出贡献
    alphas = mat(zeros((m,1)))
    iter = 0
    while (iter < maxIter):
        #记录alpha是否得到优化
        alphaPairsChanged = 0
        #遍历矩阵的第i个元素
        for i in range(m):
            #alpha(1*m)*label(m*n)=1*n
            #得知alpha向量后计算f(x)
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            #fXi为预测的类别
            Ei = fXi - float(labelMat[i])#if checks if an example violates KKT conditions
            #Ei 为预测类别和实际类别的差值
            #如果误差很大(不满足KKT条件)则进行调整,当进行
            ### check and pick up the alpha who violates the KKT condition
            ## satisfy KKT condition
            #C为常数,表示容忍度重要性
            # 1) yi*f(i) >= 1 and alpha == 0 (outside the boundary,不会影响最后的f(x))
            # 2) yi*f(i) == 1 and 0<alpha< C (on the boundary,在容忍度范围内)
            # 3) yi*f(i) <= 1 and alpha == C (between the boundary,在容忍度之外)
            ## violate KKT condition
            # because y[i]*E_i = y[i]*f(i) - y[i]^2 = y[i]*f(i) - 1, so
            # 1) if y[i]*E_i < 0, so yi*f(i) < 1, if alpha < C, violate!(alpha = C will be correct)
            # 2) if y[i]*E_i > 0, so yi*f(i) > 1, if alpha > 0, violate!(alpha = 0 will be correct)
            # 3) if y[i]*E_i = 0, so yi*f(i) = 1, it is on the boundary, needless optimized

            #对违反KKT条件的数值进行优化,首先遍历在分隔边界上的点
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                #从剩余的元素中随机选择一个j,计算其误差
                j = selectJrand(i,m)
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                #从给定的
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: print "L==H"; continue
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: print "eta>=0"; continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])#update i by the same amount as j
                #the update is in the oppostie direction
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print "iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print "iteration number: %d" % iter
    return b,alphas
