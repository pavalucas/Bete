"""
This module contains all implemented models to predict Relation Extraction (RE).
Author: Lucas Pavanelli
"""

from sklearn import svm

class SVM:
    def __init__(self):
        self.model = svm.SVC(decision_function_shape='ovr')

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class BERTForRelationExtraction:
    pass