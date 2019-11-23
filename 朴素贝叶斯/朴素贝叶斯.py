from numpy import *


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # 切分的词条
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set()
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWordsToVec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word %s is not in the vocabList" % word)
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p1Num = ones(numWords)
    p0Num = ones(numWords)
    p1Denom = 2.0
    p0Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vec = log(p1Num/p1Denom)
    p0Vec = log(p0Num/p0Denom)
    return p1Vec,p0Vec,pAbusive
def classifyNB(vec2Classify, p0Vec, p1Vec, pAbusive):
    p1 = sum(vec2Classify*p1Vec) + log(pAbusive)
    p0 = sum(vec2Classify*p0Vec) + log(1-pAbusive)
    return 1 if p1>p0 else 0



if __name__ == '__main__':
    dataSet, classVec = loadDataSet()
    vocabList = createVocabList(dataSet)
    trainMatrix = []
    for document in dataSet:
        docOfVec = setOfWordsToVec(vocabList, document)
        trainMatrix.append(docOfVec)
    p1Vec, p0Vec, pAbusive = trainNB0(trainMatrix, classVec)
    testDoc = ["love", "my", "dalmation"]
    testVec = array(setOfWordsToVec(vocabList, testDoc))
    print(",".join(testDoc)+ " classified as " + str(classifyNB(testVec, p0Vec, p1Vec, pAbusive)))

    testDoc2 = ["stupid", "garbage"]
    testVec2 = array(setOfWordsToVec(vocabList, testDoc2))
    print(",".join(testDoc2) + " classified as " + str(classifyNB(testVec2, p0Vec, p1Vec, pAbusive)))


