import numpy as np
import scipy
import scipy.io as sio
import sklearn.metrics as metrics
import csv
import matplotlib.pyplot as plt
import BinaryRandomForest
import BinaryNode
import BinaryDecisionTree

data = sio.loadmat('hw5_data/spam_data/spam_data.mat')
raw_DATA, raw_Label, test_data = data['training_data'], data['training_labels'], data['test_data']
raw_Label = raw_Label.reshape(raw_Label.shape[1])
indices1, indices2 = np.random.choice(raw_DATA.shape[0], 1000, replace=False), np.delete(np.arange(raw_DATA.shape[0]), indices1)
valid_d, valid_labels, train_d, train_labels = raw_DATA[indices1], raw_Label[indices1], raw_DATA[indices2], raw_Label[indices2]
tree = BinaryDecisionTree.BinaryDecisionTree({'depth_max': 40, 'threshold_leaf': 0.2})
tree.train(train_data, train_labels)
train_predictions = [tree.predict(datum)[0] for datum in train_d]
valid_predictions = [tree.predict(datum)[0] for datum in valid_d]
forest_params = {'num':10, 'ratio':1, 'tree_params':{'depth_max': 40, 'threshold': 0.02}, 'sampling': lambda n : np.random.choice(n, size=(int(np.sqrt(n))+1), replace=False)}
forest = BinaryRandomForest.BinaryRandomForest(forest_params)
forest.train(train_data, train_labels)
print("DecsisionTree Train accuracy: {0}".format(metrics.accuracy_score(train_labels, train_predictions)))
print("Validation Tree accuracy: {0}".format(metrics.accuracy_score(valid_labels, valid_predictions)))
print("Random Forest Train accuracy: {0}".format(metrics.accuracy_score(train_labels, [forest.predict(datum)[0] for datum in train_d])))
print("Random Forest Validation accuracy: {0}".format(metrics.accuracy_score(valid_labels, [forest.predict(datum)[0] for datum in valid_d)))