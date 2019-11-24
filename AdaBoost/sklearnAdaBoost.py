import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier

def loadData():
    dataMat = np.matrix([
        [1., 2.1],
        [2., 1.1],
        [1.3, 1.],
        [1., 1.],
        [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels

if __name__ == '__main__':
    dataMat, classLabels = loadData()
    model = AdaBoostClassifier()
    model.fit(dataMat, classLabels)
    print(model.estimator_params)
    print(model.estimator_weights_)
    print(model.feature_importances_)