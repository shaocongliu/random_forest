# shaocong liu
import BinaryNode
import BinaryDecisionTree
import numpy as np
import scipy
import scipy.io as sio
import sklearn.metrics as metrics
import csv
import matplotlib.pyplot as plt


class BinaryRandomForest:
    def __init__(self, params):
        self.num = params['num']
        self.tree_params['sampling'] = params['sampling']
        self.ratio = params['ratio']
        self.tree_params = params['tree_params']
    
    def train(self, data, labels):
        self.forest = []
        for i in range(self.num):
            tree = BinaryDecisionTree.BinaryDecisionTree(self.tree_params)
        	index = np.random.choice(data.shape[0], int(data.shape[0] * self.ratio))
            tree.train(data[index], labels[index])
            self.forest.append(tree)
    
    def predict(self, data):
        p = 0
        for tree in self.forest:
            prediction, prob = tree.predict(data)
            if (prediction != 1):
                p += 1 - prob
            else:
                p += prob
        if (p / self.num <= 0.5):
            return 0, 1 - p / self.num
        else:
            return 1, p / self.num