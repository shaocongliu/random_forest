# shaocong liu
import numpy as np
import scipy
import scipy.io as sio
import sklearn.metrics as metrics
import csv
import matplotlib.pyplot as plt

class BinaryNode:
    def __init__(self, split_rule, left, right):
        self.label = None
        self.split_rule = split_rule
        self.left = left
        self.right = right
    
    def init_leaf(self, A_count, B_count):
        if (A_count / (A_count + B_count) <= 0.5):
            self.prob = 1 - A_count / (A_count + B_count)
            self.label = 0
        else:
            self.prob = A_count / (A_count + B_count)
            self.label = 1