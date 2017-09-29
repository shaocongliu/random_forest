# shaocong liu
import BinaryRandomForest
import BinaryNode
import numpy as np
import scipy
import scipy.io as sio
import sklearn.metrics as metrics
import csv
import matplotlib.pyplot as plt

class BinaryDecisionTree:
    def __init__(self, params):
        if ('sampling' not in params):
            self.sampling = range
        else:
            self.sampling = params['sampling']
        self.depth_max = params['depth_max']
        self.threshold_leaf = params['threshold_leaf']
        self.root = None
        
    def train(self, data, labels):

        def segmenter(self, data, labels):
            minn, split, i_left, i_right, label_left, label_right = 1e50, None, None, None, None, None
            for feature in self.sampling(data.shape[1]):
                features = data[:,feature]
                for threshold in set(features):
                    left = np.argwhere(features < threshold).reshape(len(np.argwhere(features < threshold)))
                    right = np.delete(np.arange(len(features)), left)
                    if (len(left) == 0 or len(right) == 0):
                        continue
                    label_left, label_right = labels[left], labels[right]
                    left_a, right_a = sum(label_left), sum(label_right)
                    left_b, right_b = len(label_left) - left_a, len(label_right) - right_a
                    curr = ((left_a + left_b) * self.entropy(left_a, left_b) + (right_a + right_b) * self.entropy(right_a, right_b)) / ((left_a + left_b) + (right_a + right_b))
                    if (curr < minn):
                        split = (feature, threshold)
                        i_left, i_right, left_label, right_label = left, right, label_left, label_right
                        minn = curr
            if (minn != 1e50):
                sreturn split, data[i_left], left_label, data[i_right], right_label
            else:
                return None, None, None, None, None

        def helper(self, data, labels, depth):
            a, b = sum(labels), len(labels) - sum(labels)
            if depth > self.depth_max or len(labels) <= 1 or self.threshold_leaf >= self.entropy(a, b):
                leaf = BinaryNode.BinaryNode(None, None, None)
                leaf.init_leaf(a, b)
                return leaf
            split, data_left, label_left, data_right, label_right = self.segmenter(self, data, labels)
            if (split == None):
                leaf = BinaryNode.BinaryNode(None, None, None).init_leaf(a, b)
                return leaf
            return BinaryNode.BinaryNode(split, helper(self, data_left, label_left, depth+1), helper(self, data_right, label_right, depth+1))
        self.root = helper(self, data, labels, 0)

    def entropy(self, a, b):
        if (a == 0 or b == 0):
            return 0
        else:
            return - (np.log(a / (a + b)) * (a / (a + b)) + np.log(b / (a + b)) * (b / (a + b)))
    
    def predict(self, data):
        while (self.root.label == None):
            feature, threshold = self.root.split_rule
            if (data[feature] < threshold):
                self.root = self.root.left
            else:
                self.root = self.root.right
        return self.root.label, self.root.prob