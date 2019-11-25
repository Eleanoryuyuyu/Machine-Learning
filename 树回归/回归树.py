import numpy as np
import matplotlib.pyplot as plt

class treeNode():
    def __init__(self, feat, val, left, right):
        """
        :param feat: 待切分特征
        :param val: 待切分特征值
        :param left: 左子树
        :param right: 右子树
        """


def loadData(filename):
    dataMat = []
    with open(filename) as f :
        for line in f.readlines():
            curLine = line.strip().split("\t")
            dataMat.append(list(map(float, curLine)))
    return dataMat
def showData(dataSet):
    dataMat = np.transpose(np.array(dataSet))
    print(dataMat)
    x = dataMat[0]
    y = dataMat[1]
    plt.scatter(x, y, s=30, alpha=0.7)
    plt.show()

def binSplitDataSet(dataMat, feature, value):
    mat0 = dataMat[np.nonzero(dataMat[:, feature] > value)[0], :]
    mat1 = dataMat[np.nonzero(dataMat[:, feature] <= value)[0], :]
    return mat0, mat1

def regLeaf(dataSet):
    """
    :param dataSet:
    :return: 负责生成叶结点，当chooseBestSplit函数不再对数据进行切分时，调用该函数来得到叶结点的模型
              回归树中，该模型就是目标变量的均值
    """
    return np.mean(dataSet[:, -1])
def regErr(dataSet):
    """
    :param dataSet:
    :return: 在给定数据集上计算目标变量的平均误差
    """
    return np.var(dataSet[:, -1]) * len(dataSet)
def chooseBestSplit(dataSet, leafType = regLeaf, errType = regErr, ops=(1,4)):
    """
    :param dataSet: 数据集
    :param leafType: 生成叶结点函数
    :param errType: 误差函数
    :param ops: tolS 是容许误差下降值， tolN，切分的最少样本数
    :return:
    """
    tolS = ops[0]
    tolN = ops[1]
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    S = errType(dataSet)
    bestS = float('inf')
    bestIndex = 0
    bestValue = 0
    for feature in range(n - 1):
        for splitValue in set(dataSet[:, feature].T.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, feature, splitValue)
            if len(mat0) < tolN or len(mat1) < tolN:
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = feature
                bestValue = splitValue
                bestS = newS
    # 如果误差减小不大则退出
    if S - bestS < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # 如果切分出的数据集很小则退出
    if len(mat0) < tolN or len(mat1) < tolN:
        return None, leafType(dataSet)
    return bestIndex, bestValue

def createTree(dataSet, leafType = regLeaf, leafErr = regErr, ops = (1, 4)):
    bestIndex, bestValue = chooseBestSplit(dataSet, leafType, leafErr, ops)
    if bestIndex is None:
        return bestValue
    regTree = {}
    regTree["splitIndex"] = bestIndex
    regTree["splitValue"] = bestValue
    leftMat, rightMat = binSplitDataSet(dataSet, bestIndex, bestValue)
    regTree["left"] = createTree(leftMat, leafType, leafType, ops)
    regTree["right"] = createTree(rightMat, leafType, leafErr, ops)
    return regTree

if __name__ == '__main__':
    fileName = "../data/tree1.txt"
    dataArr = loadData(fileName)
    dataMat = np.mat(dataArr)
    showData(dataMat)
    regTree = createTree(dataMat)
    print(regTree)


