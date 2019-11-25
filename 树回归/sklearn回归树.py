import numpy as np
from sklearn.tree import DecisionTreeRegressor


def loadData(filename):
    dataMat = []
    with open(filename) as f :
        for line in f.readlines():
            curLine = line.strip().split("\t")
            dataMat.append(list(map(float, curLine)))
    return dataMat

if __name__ == '__main__':
    fileName = "../data/tree1.txt"
    dataArr = loadData(fileName)
    X_train = np.transpose(dataArr)[0]
    Y_train = np.transpose(dataArr)[1]
    model  = DecisionTreeRegressor()
    model.fit(X_train.reshape(-1, 1), Y_train)
