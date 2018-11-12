# -*- coding: utf-8 -*-
"""
@author: Mhamed
"""

import numpy as np
import math
from scipy.special import expit


class QDA():

    def __init__(self):

        # Assumption : Y ~ B(p)
        self.pi = None

        # Assumption x | y=1 , x | y=0 follow normal distributions with different means/covariance matrices
        self.mu_0 = None
        self.mu_1 = None
        self.sigma_0 = None
        self.sigma_1 = None

        # Parameters for P(y=1|x)
        self.w = None
        self.b = None
        self.Q = None

    def fit(self,X,y):
        """
        Finds the MLE estimates from the training data
        and calculates the coefficients for the decision boundary
        """

        n = X.shape[0] #size of sample
        N = y.sum()  #size of positive samples 
        pos = X[y == 1] #positive samples
        neg = X[y == 0] #nega as calculated theoraticallytive samples

        # MLE estimates
        self.pi = N / n
        self.mu_0 = np.sum(neg,axis=0) / (n - N)
        self.mu_1 = np.sum(pos,axis=0) / N
        self.sigma_0 = 1/(n-N) * ((neg-self.mu_0).T.dot(neg-self.mu_0))
        self.sigma_1 = 1/N * ((pos-self.mu_1).T.dot(pos-self.mu_1))
        

        # Calculate parameters for p(y=1|b) = 1/(1+exp(-(1/2 * x^T*Q*x + w^T*x + b)))
        inv_sigma_0 = np.linalg.inv(self.sigma_0)
        inv_sigma_1 = np.linalg.inv(self.sigma_1)
        self.Q = -0.5*(inv_sigma_1 - inv_sigma_0)
        self.w = self.mu_1.dot(inv_sigma_1) - self.mu_0.dot(inv_sigma_0) 
        self.b = -1/2 * (self.mu_1.T.dot(inv_sigma_1).dot(self.mu_1) - self.mu_0.T.dot(inv_sigma_0).dot(self.mu_0)) + \
                   -1/2*(math.log(np.linalg.det(self.sigma_1)) - math.log(np.linalg.det(self.sigma_0)) ) - math.log((1-self.pi)/self.pi)
        
        #self.w = inv_sigma.dot((self.mu_1 - self.mu_0).T)

    def get_params(self):
        """Get parameters for this estimator.
        
        Returns
        -------
        params : Parameter names mapped to their values.
        """
        params = {"pi":  self.pi, "mu_0": self.mu_0, "mu_1" : self.mu_1, "sigma_0" : self.sigma_0, "sigma_1" : self.sigma_1,
                  "Q": self.Q, "w" : self.w, "b":self.b}
        return params

        
    def predict_proba(self,X): 
        """ Predict p(y=1|x) for an input set X
        Returns 
        -------
        P array of probability predictions
        """
        P = [expit(np.dot(x.T,self.Q.dot(x)) + self.w.T.dot(x) + self.b) for x in X]
        return np.array(P)
    
    
    def predict(self,X,threshold = 0.5): 
        """ Predict output labels for an input set S
        Returns 
        -------
        P array of label predictions
        """
        P = self.predict_proba(X)
        P[P>=0.5] = 1
        P[P<0.5] = 0
        return P