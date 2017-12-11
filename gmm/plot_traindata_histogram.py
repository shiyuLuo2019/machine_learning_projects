#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Plots the histogram for gmm_train.csv.

@author: sherryluo
"""

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    file_path = 'gmm_train.csv'
    train_data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    X_train = train_data[:, :-1]
    Y_train = train_data[:, -1]
    X_train1 = X_train[np.where(Y_train == 1)]
    X_train2 = X_train[np.where(Y_train == 2)]

    bins = 50
    fig = plt.figure()
    
    ax = plt.subplot(2, 1, 1)
    ax.set_title('distribution of class 1')
    ax.set_xlabel('input value')
    ax.set_ylabel('frequency')
    ax.hist(X_train1, bins)
    
    ax = plt.subplot(2, 1, 2)
    ax.set_title('distribution of class 2')
    ax.set_xlabel('input value')
    ax.set_ylabel('frequency')
    ax.hist(X_train2, bins)
    
    plt.tight_layout()
    plt.savefig('class-distribution.png')
