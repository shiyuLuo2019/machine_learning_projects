#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 17:57:54 2017

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
    
    
    # how does choice for <i> affect user-based collaborative filtering?
    # k = 10, distance = 1
    flag1_errs_usercf = exp(cf=user_based_cf, tests=tests, distance=1, k=10, iFlag=1)
    flag0_errs_usercf = exp(cf=user_based_cf, tests=tests, distance=1, k=10, iFlag=0)
    t, p = paired_t(flag1_errs_usercf, flag0_errs_usercf)
    plot(flag1_errs_usercf, flag0_errs_usercf, 'iFlag = 1', 'iFlag = 0', 'prob4d1.png')
    with open('log.txt', 'a') as f:
        f.write('experiment d1:\n')
        f.write('user based cf, manhattan distance, k = 10, iFlag = 1')
        np.savetxt(f, flag1_errs_usercf, newline='\t')
        f.write('\nuser based cf, manhattan distance, k = 10, iFlag = 0')
        np.savetxt(f, flag0_errs_usercf, newline='\t')
        f.write('\nt = {}, p = {}\n'.format(t, p))
        f.write('plot saved as probe4d1.png\n\n')
    best_iflag = np.argmin([np.mean(flag0_errs_usercf), np.mean(flag1_errs_usercf)])
    with open('best-iflag.txt', 'w') as f:
        f.write(str(best_iflag))
  
    
main()