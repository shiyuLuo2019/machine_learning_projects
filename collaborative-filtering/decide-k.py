#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 17:59:17 2017

@author: sherryluo
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 18:09:49 2017

@author: sherryluo
"""
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import math
from user_cf import user_based_cf
from item_cf import item_based_cf

num_samples = 50
sample_size = 100
numOfUsers = 943
numOfItems = 1682

f = open('log.txt', 'a')
f.close()
    
def measure_error(trueRating, predictedRating):
    return math.fabs(trueRating - predictedRating)


def exp(cf, tests, distance, k, iFlag):
    # hold 50 sample errors
    sample_errors = np.zeros(num_samples)
    # compute average errors of each sample (test-prior pair)
    for i in range(num_samples):
        # prior set of sample i
        fname = 'prior{}.txt'.format(i)
        # test set of sample i
        test = tests[i:i+sample_size]
        # initialize error
        error = 0
        # measure error for each rating in sample i
        for j in range(sample_size):
            # userid of j-th rating in sample i
            userid = test[j][0]
            # movieid of j-th rating in sample i
            movieid = test[j][1]
            trueRating, predictedRating = cf(fname, userid, movieid, distance, k, iFlag, numOfUsers, numOfItems)
            trueRating = test[j][2]
            error = error + measure_error(trueRating, predictedRating)
        # average error and add to error_list
        error = float(error) / sample_size
        sample_errors[i] = error
    return sample_errors


def plot(sample_errors_1, sample_errors_2, label1, label2, figname):
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(sample_errors_1, label=label1)
    plt.plot(sample_errors_2, label=label2)
    plt.legend(loc='upper left')
    plt.savefig(figname)
 

def paired_t(sample1, sample2):
    result = ttest_rel(sample1, sample2)
    t = result[0]
    p = result[1]
    return t, p
    
    
def main():
    # load 50 tests
    tests = np.loadtxt('tests.txt', dtype=int)
    
    # use the best settings decided and varies k
    # what is the best k for user-based collaborative filtering?
    # simply choose the one that results in lowest error
    with open('best-dist.txt', 'r') as f:
        best_dist = f.readline()
        best_dist = int(best_dist)
    with open('best-iflag.txt', 'r') as f:
        best_iflag = f.readline()
        best_iflag = int(best_iflag)
        
    k = 1
    k_sample_errors = np.zeros(6)
    for i in range(6):
        sample_errors = exp(cf=user_based_cf, tests=tests, distance=best_dist, k=k, iFlag=best_iflag)
        k_sample_errors[i] = np.mean(sample_errors)
        k = k * 2
    # plot
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot([1, 2, 4, 8, 16, 32], k_sample_errors)
    plt.savefig('probe.png')
    
    ind_best_k = np.argmin(k_sample_errors)
    best_k = int(math.pow(2, ind_best_k))
    with open('best-k.txt', 'w') as f:
        f.write(str(best_k))
    

    
main()