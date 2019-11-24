from numpy import *
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LogisticRegression

def loadData():
    dataMat = []
    labelMat = []
    with open ("../data/testSet.txt") as f:
        for line in f.readlines():
            lineArr = line.strip().split()
            dataMat.append([1.0, float(lineArr[0]),float(lineArr[1])]) #1.0 为bias项， w0*x0 = w0
            labelMat.append(int(lineArr[2]))
    return dataMat, labelMat
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
    trainMat, labelMat = loadData()
    lr_model = LogisticRegression(solver='liblinear', penalty='l2',max_iter=5000)
    lr_model.fit(trainMat, labelMat)
    weights = lr_model.coef_.reshape(-1, 1)
    print(weights)
    plotBestFit(weights)

