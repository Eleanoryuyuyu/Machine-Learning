from numpy import *
import matplotlib.pyplot as plt
import random
def loadData():
    dataMat = []
    labelMat = []
    with open ("../data/testSet.txt") as f:
        for line in f.readlines():
            lineArr = line.strip().split()
            dataMat.append([1.0, float(lineArr[0]),float(lineArr[1])]) #1.0 为bias项， w0*x0 = w0
            labelMat.append(int(lineArr[2]))
    return dataMat, labelMat
def sigmoid(inX):
    return 1.0/(1 + exp(-inX))
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error # Logistic 梯度公式推导
    return weights

def stocGrandAscent(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    numIter = 150
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(dataMatrix[randIndex]*weights)
            error = classLabels[randIndex] - h
            weights += alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights
def plotBestFit(wei):
    #weights = wei.getA()
    weights = wei
    dataMat, labelMat = loadData()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if labelMat[i] == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = arange(-3.0,3.0,0.1)
    y = (-weights[0] - weights[1]*x)/weights[2] # 设定 0 = w0x0 + w1x1 + w2x2 然后解出x1和x2的关系式
    ax.plot(x,y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

if __name__ == '__main__':
    dataMat, labelMat = loadData()
    # weights = gradAscent(dataMat, labelMat)
    weights = stocGrandAscent(dataMat, labelMat)
    print(weights)
    plotBestFit(weights)

