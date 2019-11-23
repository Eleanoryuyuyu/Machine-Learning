from math import log
from operator import itemgetter


def calShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for data in dataSet:
        if data[-1] not in labelCounts:
            labelCounts[data[-1]] = 0
        labelCounts[data[-1]] += 1
    shannonEnt = 0.0
    for label in labelCounts.keys():
        prob = float(labelCounts[label]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataSet(dataSet, index, value):
    retDataSet = []
    for data in dataSet:
        if data[index] == value:
            reducedData = data[:index]
            reducedData.extend(data[index + 1:])
            retDataSet.append(reducedData)
    return retDataSet


def chooseBeatFeatureToSplit(dataSet):
    numFeatures = len([dataSet[0]]) - 1
    baseEntropy = calShannonEnt(dataSet)
    beatInfoGain = 0.0
    beatFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        newEntropy = 0.0
        featSet = set(featList)
        for feat in featSet:
            retData = splitDataSet(dataSet, i, feat)
            prob = float(len(retData)) / len(dataSet)
            newEntropy += prob * calShannonEnt(retData)
        infoGain = baseEntropy - newEntropy
        if infoGain > beatInfoGain:
            beatInfoGain = infoGain
            beatFeature = i
    return beatFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount:
            classCount[vote] = 0
        classCount[vote] += 1
    classCount = sorted(classCount.items(), key=itemgetter(1), reverse=True)
    return classCount[0][0]

def createTree(dataSet,features, bestFeatures):
    labelList = [example[-1] for example in dataSet]
    if labelList.count(labelList[0]) == len(dataSet):
        return labelList[0]
    if len(dataSet[0]) == 1 or len(bestFeatures) == 0:
        return majorityCnt(labelList[0])
    bestFeat = chooseBeatFeatureToSplit(dataSet)
    bestFeatureName = features[bestFeat]
    bestFeatures.append(bestFeatureName)
    myTree = {bestFeatureName:{}}
    del (features[bestFeat])
    featureValues = [example[bestFeat] for example in dataSet]
    uniqueValues = set(featureValues)
    for value in uniqueValues:
        subFeatures = features[:]
        myTree[bestFeatureName][value] = createTree(splitDataSet(dataSet, bestFeat, value), subFeatures, bestFeatures)
    return myTree


