from sklearn.naive_bayes import MultinomialNB
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

if __name__ == '__main__':
    dataSet, classVec = loadDataSet()
    vocabList = createVocabList(dataSet)
    trainMatrix = []
    for document in dataSet:
        docOfVec = setOfWordsToVec(vocabList, document)
        trainMatrix.append(docOfVec)
    nbClassify = MultinomialNB()
    nbClassify.fit(trainMatrix, classVec)
    testDoc = ["love", "my", "dalmation"]
    testVec = array(setOfWordsToVec(vocabList, testDoc)).reshape(1, -1)
    predict = nbClassify.predict(testVec)
    print(",".join(testDoc)+ " classified as " + str(predict))

    testDoc2 = ["stupid", "garbage"]
    testVec2 = array(setOfWordsToVec(vocabList, testDoc2)).reshape(1, -1)
    predict2 = nbClassify.predict(testVec2)
    print(",".join(testDoc2) + " classified as " + str(predict2))
