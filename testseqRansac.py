# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 13:03:54 2018

@author: Sikasjc
"""

import numpy as np
import matplotlib.pyplot as plt

import random
import Ransac
import time

num_iterations = 500
num_samples = 1000
noise_ratio = 0.8
num_noise = int(noise_ratio * num_samples)

def setup():
    global ax1
    num = 2
    X = np.asarray(range(num_samples))
    Y1 = 1 * X
    Y2 = 2 * X
    Y = np.asarray([Y1[i] if i % 2!=0 else Y2[i] for i in range(num_samples)])
    noise = [random.randint(0, 2 * (num_samples - 1)) for i in range(num_noise)]
    Y[random.sample(range(len(Y)), num_noise)] = noise
    data = np.asarray([X, Y]).T
    model = Ransac.LineModel(num)
    ax1 = plt.subplot(1,2,1)
    plt.plot(X, Y, 'bx')
    return data, model

def run(data, model):
    random_seed = random.randint(0,100)
    start_time = time.time()
    (params, inliers, residuals, iterations) = Ransac.seqRansac(data, model, 2, 0.08, num_iterations, 1e-10, random_seed)
    end_time = time.time()
    mean_time = (end_time - start_time) / num_iterations
    return params, residuals, mean_time, iterations
    
    
def summary(params, residual, mean_time, iterations):
    print(" Paramters ".center(40, '='))
    print(params)
    print(" Residual ".center(40, '='))
    print(residual)
    print(" Iterations ".center(40, '='))
    print(iterations)
    print(" Time ".center(40, '='))
    print("%.1f msecs mean time spent per call" % (1000 * mean_time))
    X = np.asanyarray([0, num_samples - 1])
    plt.subplot(1,2,2,sharey=ax1)
    for param in params:
        plt.plot(X, param[0] * X + param[1], 'y-')
        
    plt.show()
    
if __name__ == '__main__':
    data, model = setup()
    try:
        params, residuals, mean_time, iterations = run(data, model)
        summary(params, residuals, mean_time, iterations)
    except ValueError as e:
       print(e) 
    