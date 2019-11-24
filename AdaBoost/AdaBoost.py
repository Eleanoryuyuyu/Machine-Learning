import numpy as np
import matplotlib.pyplot as plt


def loadData():
    dataMat = np.matrix([
        [1., 2.1],
        [2., 1.1],
        [1.3, 1.],
        [1., 1.],
        [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels


def showData(dataMat, labelMat):
    data_plus = []
    data_minus = []
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    # print(np.shape(data_plus_np))
    # print(np.shape(np.transpose(data_plus_np)))
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1])
    plt.show()


# 基分类器
def stumpClassify(dataMat, dimen, threshVal, threshIneq):
    retArr = np.ones((len(dataMat), 1))
    if threshIneq == 'lt':
        retArr[dataMat[:, dimen] < threshVal] = -1.0
    else:
        retArr[dataMat[:, dimen] > threshVal] = -1.0
    return retArr


def buildStump(dataArr, classLabels, D):
    dataMat = np.mat(dataArr)
    labelMat = np.mat(classLabels)
    m, n = np.shape(dataMat)
    numSteps = 10
    bestStump = {}
    bestClasEat = np.mat(np.zeros((m, 1)))
    minError = float('inf')
    for i in range(n):
        rangeMin = dataMat[:, i].min()
        rangeMax = dataMat[:, i].max()
        stepSize = (rangeMax - rangeMin) // numSteps
        for j in range(-1, int(stepSize) + 1):
            for inequa in ['lt', 'gt']:
                threshVal = rangeMin + float(j) + stepSize
                predictVals = stumpClassify(dataMat, i, threshVal, inequa)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictVals.reshape(-1, 1) == labelMat.reshape(-1,1)] = 0
                weightedErr = D.T * errArr
                if weightedErr < minError:
                    minError = weightedErr
                    bestClasEat = predictVals.copy()
                    bestStump['dim'] = i
                    bestStump['threshVal'] = threshVal
                    bestStump['ineq'] = inequa
    return bestStump, minError, bestClasEat

def adaBoostTrainDS(dataArr, classLabels, numIter = 40):
    weakClassArr = []
    m = len(dataMat)
    D = np.mat(np.ones((m,1))/m)
    aggClassEst = np.mat(np.zeros((m, 1)))
    for iter in range(numIter):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print("D:", D.T)
        alpha = float(0.5*np.log(1.0 - error) / max(error, 1e-16))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        expon = np.multiply(-1*alpha*np.mat(classLabels).T, classEst )
        D = np.multiply(D ,np.exp(expon))
        D = D/D.sum()
        aggClassEst  += alpha * classEst
        print("classEst:", aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels), np.ones((m, 1)))
        errorRate  =aggErrors.sum()/m
        print("errorRate :", errorRate)
        if errorRate == 0.0:
            break
    return weakClassArr


if __name__ == '__main__':
    dataMat, labelMat = loadData()
    showData(dataMat, labelMat)
    weakClassify = adaBoostTrainDS(dataMat, labelMat)
    print(weakClassify)