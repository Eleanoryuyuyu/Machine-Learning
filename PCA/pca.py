import numpy as np
import matplotlib.pyplot as plt


def loadData():
    dataArr = []
    with open("../data/pca.txt") as f:
        for line in f.readlines():
            lineArr = line.strip().split("\t")
            dataArr.append(list(map(float, lineArr)))
    return dataArr

def showData(dataArr):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.transpose(dataArr)[0]
    y = np.transpose(dataArr)[1]
    ax.scatter(x, y, marker='^',s=30, c='blue')
    lowDDataMat, newDataMat = pca(dataArr, 1)
    print(np.transpose(newDataMat))
    x1 = np.transpose(np.array(newDataMat))[0]
    y1 = np.transpose(np.array(newDataMat))[1]
    ax.scatter(x1, y1, marker='o', s=80, c='red')
    plt.show()

def pca(dataMat, topNfeat=9999999):
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    covMat = np.cov(meanRemoved, rowvar=False)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    eigValIndex = np.argsort(eigVals)
    eigValIndex = eigValIndex[:-(topNfeat+1):-1]
    redEigVecs = eigVects[:, eigValIndex]
    lowDDataMat = meanRemoved * redEigVecs
    newDataMat = lowDDataMat * redEigVecs.T + meanVals
    return lowDDataMat,newDataMat



if __name__ == '__main__':
    dataArr = loadData()
    showData(dataArr)