# -*- coding: utf-8 -*-
"""
@author: Mhamed
"""

import numpy as np
from scipy.special import expit

class LogisticRegression():
    
    def __init__(self):
        self.w = None
        self.b = None
        
    def fit(self, X,y, nb_iter=20):
        # Include a vector of one as a feature to get the constant term
        X2 = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        # Initialize the coefficients before starting the Newton updates
        w = np.random.rand(X2.shape[1]) * 0.001
        
        for i in range(nb_iter):      
            eta = expit(X2.dot(w)) #using expit instead of self-coded sigmoid allows us to easily avoid OverFlow errors            
            gradient = X2.T.dot(y - eta)
            hessian = -X2.T.dot(np.diag(eta*(1-eta))).dot(X2)
            search_dir = np.linalg.inv(hessian).dot(gradient)
            w = w - search_dir
        
        self.w, self.b = w[0:-1], w[-1]
        
    def get_params(self):
        params = {"w" : self.w, "b":self.b}
        return params

    def predict_proba(self, X): 
        """ Predict p(y=1|x) for an input set X
        Returns 
        -------
        P array of probability predictions
        """
        P = expit(X.dot(self.w)+self.b)
        return P
    
    def predict(self, X): 
        P = self.predict_proba(X)
        P[P>0.5] = 1
        P[P<=0.5] = 0
        return P
    

