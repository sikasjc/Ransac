# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 13:27:55 2018

@author: Sikasjc
"""

import numpy as np
import random

class Model(object):
    def fit(self, data):
        raise NotImplementedError
        
    def distance(self, data):
        raise NotImplementedError

class LineModel(Model):
    """
    A 2D line model.
    """
    def __init__(self, num=None):
        self.params = None
        self.dists = None
        self.num = num
        
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
        self.params = [k, m]
        self.residual = sum(abs(k * X + m - Y))
    
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

def ransac(data, model, min_samples, min_inliers, iterations=100, eps=1e-10, random_seed=42):
    """
    Fits a model to observed data.
    
    Uses the RANSC iterative method of fitting a model to observed data.
    """
    random.seed(random_seed)
    
    if len(data) <= min_samples:
        raise ValueError("Not enough input data to fit the model.")
        
    if 0 < min_inliers and min_inliers < 1:
        min_inliers = int(min_inliers * len(data))
        
    best_params = None
    best_inliers = None
    best_residual = np.inf
    best_iteration = None
    
    for i in range(iterations):
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
                    best_iteration = i
                    
        except ZeroDivisionError as e:
            print(e)
        
    if best_params is None:
        raise ValueError("RANSAC failed to find a sufficiently good fit for the data.")
    else:
        return (best_params, best_inliers, best_residual, best_iteration)
    
def seqRansac(data, model, min_samples, min_inliers, num_iterations=100, eps=1e-10, random_seed=42):
    """
    Fits a model to observed data.
    
    Uses the Sequential RANSC iterative method of fitting a model to observed data.
    """
    random.seed(random_seed)
    
    if len(data) <= min_samples:
        raise ValueError("Not enough input data to fit the model.")
        
    if 0 < min_inliers and min_inliers < 1:
        min_inliers = int(min_inliers * len(data))
        
    params = []
    inliers = []
    residuals = []
    iterations = []
    
    for i in range(model.num):
    
        best_params = None
        best_inliers = None
        best_residual = np.inf
        best_iteration = None
        
        for i in range(num_iterations):
            _indices = list(range(len(data)))
            random.shuffle(_indices)
            _inliers = np.asarray([data[i] for i in _indices[:min_samples]])
            shuffled_data = np.asarray([data[i] for i in _indices[min_samples:]])
            
            try:
                model.fit(_inliers)
                dists = model.distance(shuffled_data)
                more_inliers = shuffled_data[np.where(dists <= eps)[0]]
                _inliers = np.concatenate((_inliers, more_inliers))
                
                if len(_inliers) >= min_inliers:
                    model.fit(_inliers)
                    if model.residual < best_residual:
                        best_params = model.params
                        best_inliers = _inliers
                        best_residual = model.residual
                        best_iteration = i
                        
            except ZeroDivisionError as e:
                print(e)
     
        if best_params:
            params.append(best_params)
            inliers.append(best_inliers)
            residuals.append(best_residual)
            iterations.append(best_iteration)
            data = np.delete(data, [i for i in best_inliers], axis=0)        
        
    if params is None:
        raise ValueError("Sequential RANSAC failed to find a sufficiently good fit for the data.")
    else:      
        return (params, inliers, residuals, iterations)