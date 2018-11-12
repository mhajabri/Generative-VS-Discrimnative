# -*- coding: utf-8 -*-
"""
@author: Mhamed
"""

import numpy as np
import math
from scipy.special import expit



class LDA():

    def __init__(self):

        # Assumption : Y ~ B(p)
        self.pi = None

        # Assumption x | y=1 , x | y=0 follow normal distributions with same covariance matrix
        self.mu_0 = None
        self.mu_1 = None
        self.sigma = None

        # Parameters for P(y=1|x)
        self.w = None
        self.b = None

    def fit(self,X,y):
        """
        Finds the MLE estimates from the training data
        and calculates the coefficients for the decision boundary
        """

        n = X.shape[0] #size of sample
        N = y.sum()  #size of positive samples 
        pos = X[y == 1] #positive samples
        neg = X[y == 0] #negative samples

        # MLE estimates
        self.pi = N / n
        self.mu_0 = np.sum(neg,axis=0) / (n - N)
        self.mu_1 = np.sum(pos,axis=0) / N
        self.sigma = 1/n * ((pos-self.mu_1).T.dot(pos-self.mu_1) + (neg-self.mu_0).T.dot(neg-self.mu_0))

        # Calculate parameters for p(y=1|b) = 1/(1+exp(-(w^T*x + b)))
        inv_sigma = np.linalg.inv(self.sigma)
        self.w = inv_sigma.dot((self.mu_1 - self.mu_0).T)
        self.b = 1/2* (self.mu_0.T.dot(inv_sigma).dot(self.mu_0) - self.mu_1.T.dot(inv_sigma).dot(self.mu_1)) -math.log((1-self.pi)/self.pi)

    def get_params(self):
        """Get parameters for this estimator.
        
        Returns
        -------
        params : Parameter names mapped to their values.
        """
        params = {"pi":  self.pi, "mu_0": self.mu_0, "mu_1" : self.mu_1, "sigma" : self.sigma,
                  "w" : self.w, "b":self.b}
        return params

        
    def predict_proba(self,X): 
        """ Predict p(y=1|x) for an input set X
        Returns 
        -------
        P array of probability predictions
        """
        P = expit(X.dot(self.w)+self.b)
        return P
    
    def predict(self,X,threshold = 0.5): 
        """ Predict output labels for an input set S
        Returns 
        -------
        P array of label predictions
        """
        P = self.predict_proba(X)
        P[P>threshold] = 1
        P[P<=threshold] = 0
        return P