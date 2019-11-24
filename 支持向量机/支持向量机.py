import random
import numpy as np
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

"""
parameter:
dataMatIn : 数据集
classLabels: 类别标签
C: 松弛变量，常数
toler: 容错率
maxIter: 最大迭代次数

"""
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMat = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    b = 0
    m, n = np.shape(dataMat)
    alphas = np.mat(np.zeros((m,1)))
    iter = 0
    while iter < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            fxi = float(np.multiply(alphas, labelMat).T * (dataMat*dataMat[i, :].T)) + b
            Ei = fxi - float(labelMat[i])
            if (labelMat[i]*Ei < -toler and alphas[i] < C) or (labelMat[i]*Ei > toler and alphas[i] > 0):
                j = selectJrand(i , m)
                fxj = float(np.multiply(alphas, labelMat).T * (dataMat*dataMat[j, :].T)) + b
                Ej = fxj  - labelMat[j]
                # 保存更新前alpha的值， 使用深拷贝
                alphaIOld = alphas[i].copy()
                alphaJOld = alphas[j].copy()
                # 步骤2 计算上下界 L和 H
                if labelMat[i] != labelMat[j]:
                    L  = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0,alphas[j]-alphas[i]-C)
                    H = min(C,alphas[j] + alphas[i])
                if L == H:
                    print("L==H")
                    continue
                # 步骤3 计算eta
                eta = 2.0 * dataMat[i, :] * dataMat[j, :].T - dataMat[i, :]*dataMat[i, :].T - dataMat[j, :] * dataMat[j, :].T
                if eta > 0:
                    print("eta > 0")
                    continue
                # 步骤4 ： 更新alpha_j
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                # 步骤5 ： 修建alpha_j
                alphas[j]  = clipAlpha(alphas[j], H, L)
                # 步骤6 ： 更新alpha_i
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJOld - alphas[j])






