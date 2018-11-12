# -*- coding: utf-8 -*-
"""
@author: Mhamed
"""

import numpy as np

class LinearRegression():
    
    def __init__(self):
        self.w = None
        self.b = None
        
    def fit(self, X, y):
        # Include a vector of one as a feature to get the constant term
        X2 = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        # the bias b term is included in W
        W = np.linalg.inv(X2.T.dot(X2)).dot(X2.T).dot(y)
        self.w, self.b = W[0:-1], W[-1]
        
    def get_params(self):
        params = {"w" : self.w, "b":self.b}
        return params

    def predict(self, X): 
        P = X.dot(self.w) + self.b
        P[P>0.5] = 1
        P[P<=0.5] = 0
        return P
