import random

def loadData():
    dataMat = []
    labelMat = []
    with open('../data/SVMTestSet.txt') as f:
        for line in f.readlines():
            lineArr = line.strip().split()
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
    return dataMat, labelMat
def selectJrand(i, m): # i 是alpha的下标，m是所有alpha的数目
    j = i
    while j==i:
        j = int(random.uniform(0, m))
    return j
def clipAlpha(aj, H, L): # 用于调整大于H或者小于L的alpha的值
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj



