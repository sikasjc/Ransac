# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 21:48:58 2018

@author: Sikasjc
"""
import numpy as np
import matplotlib.pyplot as plt

import random
import Ransac
import time

num_iterations = 500
num_samples = 2000
noise_ratio = 0.9
num_noise = int(noise_ratio * num_samples)

def setup():
    global ax1
    X = np.asarray(range(num_samples))
    Y = 0.5 * X
    noise = [random.randint(0, 2 * (num_samples - 1)) for i in range(num_noise)]
    Y[random.sample(range(len(Y)), num_noise)] = noise
    data = np.asarray([X, Y]).T
    model = Ransac.LineModel()
    plt.figure(figsize=(10,5))
    ax1 = plt.subplot(1,2,1)
    plt.plot(X, Y, 'bx')
    
    return data, model

def run(data, model):
    random_seed = random.randint(0,100)
    start_time = time.time()
    (params, inliers, residual, iterations) = Ransac.ransac(data, model, 2, (1 - noise_ratio) * num_samples, num_iterations, eps=0.1, random_seed=random_seed)
    end_time = time.time()
    mean_time = (end_time - start_time) / num_iterations
    return params, residual, mean_time, iterations
    
    
def summary(params, residual, mean_time, iterations):
    print(" Paramters ".center(40, '='))
    print(params)
    print(" Residual ".center(40, '='))
    print(residual)
    print(" Iteration ".center(40, '='))
    print(iterations, "iteration finds the best params")
    print(" Time ".center(40, '='))
    print("%.1f msecs mean time spent per iteration" % (1000 * mean_time))
    X = np.asanyarray([0, num_samples - 1])
    Y = params[0] * X + params[1]
    plt.subplot(1,2,2, sharey = ax1)
    plt.plot(X, Y, 'g-', linewidth=2)
    
    plt.show()
    
if __name__ == '__main__':
    data, model = setup()
    try:
        params, residual, mean_time, iterations = run(data, model)
        summary(params, residual, mean_time, iterations)
    except ValueError as e:
       print(e) 
    