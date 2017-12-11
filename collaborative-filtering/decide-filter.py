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
    
    # decide best filter
    with open('best-dist.txt', 'r') as f:
        best_dist = f.readline()
        best_dist = int(best_dist)
    with open('best-iflag.txt', 'r') as f:
        best_iflag = f.readline()
        best_iflag = int(best_iflag)
    with open('best-k.txt', 'r') as f:
        best_k = f.readline()
        best_k = int(best_k)
    
    errs_usercf = exp(cf=user_based_cf, tests=tests, distance=best_dist, k=best_k, iFlag=best_iflag)
    errs_itemcf = exp(cf=item_based_cf, tests=tests, distance=best_dist, k=best_k, iFlag=best_iflag)
    plot(errs_usercf, errs_itemcf, 'user cf', 'item cf', 'probf.png')
    t_filter, p_filter = paired_t(errs_usercf, errs_itemcf)
    avg_err_usercf = np.mean(errs_usercf)
    avg_err_itemcf = np.mean(errs_itemcf)
    
    with open('best-filter.txt', 'w') as f:
        f.write('user cf:\n')
        np.savetxt(f, errs_usercf, newline='\t')
        f.write('average: {}\n'.format(avg_err_usercf))
        f.write('item cf:\n')
        np.savetxt(f, errs_itemcf, newline='\t')
        f.write('\naverage: {}\n'.format(avg_err_itemcf))
        f.write('t value: {}, p value: {}'.format(t_filter, p_filter))

main()
