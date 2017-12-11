#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 17:56:19 2017

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

f = open('log.txt', 'w')
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
    
    # how does <distance> affect item-based collaborative filtering?
    # k = 10, iFlag = 0
    manhattan_errs_itemcf = exp(cf=item_based_cf, tests=tests, distance=1, k=10, iFlag=0)
    pearson_errs_itemcf = exp(cf=item_based_cf, tests=tests, distance=0, k=10, iFlag=0)
    t, p = paired_t(manhattan_errs_itemcf, pearson_errs_itemcf)
    plot(manhattan_errs_itemcf, pearson_errs_itemcf, 'manhattan', 'pearson', 'prob4c1.png')
    with open('log.txt', 'a') as f:
        f.write('experiment c1:\n')
        f.write('item based cf, manhattan distance, k = 10, iFlag = 0\n')
        np.savetxt(f, manhattan_errs_itemcf, newline='\t')
        f.write('\nitem based cf, pearson correlation, k = 10, iFlag = 0\n')
        np.savetxt(f, pearson_errs_itemcf, newline='\t')
        f.write('\nt = {}, p = {}\n'.format(t, p))
        f.write('plot saved as prob4c1.png\n\n')
    best_dist = np.argmin([np.mean(pearson_errs_itemcf), np.mean(manhattan_errs_itemcf)])
    with open('best-dist.txt', 'w') as f:
        f.write(str(best_dist))

main()