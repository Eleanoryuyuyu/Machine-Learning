import random
import numpy as np
import matplotlib.pyplot as plt


def loadData():
    dataMat = []
    labelMat = []
    with open('../data/SVMTestSet.txt') as f:
        for line in f.readlines():
            lineArr = line.strip().split()
            dataMat.append([float(lineArr[0]), float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def selectJrand(i, m):  # i 是alpha的下标，m是所有alpha的数目
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):  # 用于调整大于H或者小于L的alpha的值
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
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0
    while iter < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            # fx = wx + b
            # w = alpha * y * x
            # fx = alpha * y * x * x + b
            fxi = float(np.multiply(alphas, labelMat).T * (dataMat * dataMat[i, :].T)) + b
            Ei = fxi - float(labelMat[i])
            # 如果真实结果和预测结果差别较大（正间隔和负间隔都会被测试）， 同时保证alpha不能等于0和C，
            # 则alpha还有优化的空间
            if (labelMat[i] * Ei < -toler and alphas[i] < C) or (labelMat[i] * Ei > toler and alphas[i] > 0):
                j = selectJrand(i, m)
                fxj = float(np.multiply(alphas, labelMat).T * (dataMat * dataMat[j, :].T)) + b
                Ej = fxj - labelMat[j]
                # 保存更新前alpha的值， 使用深拷贝
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # 步骤2 计算上下界 L和 H
                if labelMat[i] != labelMat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] - alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L==H")
                    continue
                # 步骤3 计算eta
                eta = 2.0 * dataMat[i, :] * dataMat[j, :].T - dataMat[i, :] * dataMat[i, :].T - dataMat[j, :] * dataMat[
                                                                                                                j, :].T
                if eta > 0:
                    print("eta > 0")
                    continue
                # 步骤4 ： 更新alpha_j
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                # 步骤5 ： 修建alpha_j
                alphas[j] = clipAlpha(alphas[j], H, L)
                # 步骤6 ： 更新alpha_i
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                # 步骤7：更新b_1和b_2
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMat[i, :] * dataMat[i, :].T - labelMat[
                    j] * (alphas[j] - alphaJold) * dataMat[i, :] * dataMat[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMat[i, :] * dataMat[j, :].T - labelMat[
                    j] * (alphas[j] - alphaJold) * dataMat[j, :] * dataMat[j, :].T
                # 步骤8：根据b_1和b_2更新b
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                # 统计优化次数
                alphaPairsChanged += 1
                # 打印统计信息
                print("第%d次迭代 样本:%d, alpha优化次数:%d" % (iter, i, alphaPairsChanged))
                # 更新迭代次数
            if (alphaPairsChanged == 0):
                iter += 1
            else:
                iter = 0
            print("迭代次数: %d" % iter)
        return b, alphas


def showClassifer(dataMat, w, b):
    # 绘制样本点
    data_plus = []  # 正样本
    data_minus = []  # 负样本
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)  # 转换为numpy矩阵
    data_minus_np = np.array(data_minus)  # 转换为numpy矩阵
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)  # 正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7)  # 负样本散点图
    # 绘制直线
    x1 = max(dataMat)[0]
    x2 = min(dataMat)[0]
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
    plt.plot([x1, x2], [y1, y2])
    # 找出支持向量点
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 0:
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
    plt.show()


def get_w(dataMat, labelMat, alphas):
    alphas, dataMat, labelMat = np.array(alphas), np.array(dataMat), np.array(labelMat)
    w = np.dot((np.tile(labelMat.reshape(1, -1).T, (1, 2)) * dataMat).T, alphas)
    return w.tolist()


if __name__ == '__main__':
    dataMat, labelMat = loadData()
    b, alphas = smoSimple(dataMat, labelMat, 0.6, 0.001, 40)
    w = get_w(dataMat, labelMat, alphas)
    showClassifer(dataMat, w, b)
