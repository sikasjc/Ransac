# -*- coding: utf-8 -*-
"""
Created on Fri May  4 00:31:43 2018

@author: Sikasjc
"""
import numpy as np
import random, sys

class Model(object):
    def fit(self, data):
        raise NotImplementedError
        
    def distance(self, data):
        raise NotImplementedError

class LineModel(Model):
    """
    A 2D line model.
    """
    def __init__(self):
        self.params = None
        self.dists = None
        
    def fit(self, data):
        """
        Fits the model to the data, minimizing the sum of absolute errors.
        """
        X = data[:,0]
        Y = data[:,1]
#        denom = (X[-1] - X[0])
#        if denom == 0:
#            raise ZeroDivisionError
#        k = (Y[-1] - Y[0]) / denom
#        m = Y[0] - k * X[0]
        A = np.vstack([X, np.ones(len(X))]).T
        k, m = np.linalg.lstsq(A, Y, None)[0]
        X_ = np.mean(X)
        Y_ = np.mean(Y)
        self.params = [k, m]
        self.residual = abs(k * X_ + m - Y_)
    
    def distance(self, samples):
        """
        Calculates the vertical distances from the samples to the model.
        """
        X = samples[:,0]
        Y = samples[:,1]
        k = self.params[0]
        m = self.params[1]
        dists = abs(k * X + m - Y)
#        dists = abs(-k * X + Y - m) / math.sqrt(k**2 + 1)
        
        return dists

def ransac_APC(data, model, min_samples, min_inliers, eps=1e-10, P = 0.99, random_seed=42):
    """
    Fits a model to observed data.
    
    Uses the RANSC iterative method of fitting a model to observed data.
    """
    random.seed(random_seed)
    
    if len(data) <= min_samples:
        raise ValueError("Not enough input data to fit the model.")
        
    iterations = sys.maxsize
    count = 0
    min_inliers = 0
    
    best_params = None
    best_inliers = None
    best_residual = np.inf
    
    while iterations > count: 
        if 0 < min_inliers and min_inliers < 1:
            min_inliers = int(min_inliers * len(data))
        
        indices = list(range(len(data)))
        random.shuffle(indices)
        inliers = np.asarray([data[i] for i in indices[:min_samples]])
        shuffled_data = np.asarray([data[i] for i in indices[min_samples:]])
        
        try:
            model.fit(inliers)
            dists = model.distance(shuffled_data)
            more_inliers = shuffled_data[np.where(dists <= eps)[0]]
            inliers = np.concatenate((inliers, more_inliers))
            
            if len(inliers) >= min_inliers:
                model.fit(inliers)
                if model.residual < best_residual:
                    best_params = model.params
                    best_inliers = inliers
                    best_residual = model.residual
                                        
        except ZeroDivisionError as e:
            print(e)
            
        ratio = len(inliers) / len(data)
        if min_inliers/len(data) < ratio:
            best_residual = np.inf
            min_inliers = ratio
            iterations = int(np.log(1 - P) / np.log(1 - min_inliers**min_samples)) + 1 
        count += 1 
        
    if best_params is None:
        raise ValueError("RANSAC failed to find a sufficiently good fit for the data.")
    else:
        return (best_params, best_inliers, best_residual, count)
