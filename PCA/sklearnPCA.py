import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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
    pca = PCA()
    pca.fit(dataArr)
    newDataMat = pca.fit_transform(dataArr)
    print(np.transpose(newDataMat))
    x1 = np.transpose(np.array(newDataMat))[0]
    y1 = np.transpose(np.array(newDataMat))[1]
    ax.scatter(x1, y1, marker='o', s=80, c='red')
    plt.show()
if __name__ == '__main__':
    dataArr = loadData()
    showData(dataArr)
