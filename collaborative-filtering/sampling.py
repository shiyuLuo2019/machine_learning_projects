#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 14:03:58 2017

@author: sherryluo
"""

import numpy as np

num_samples = 50
sample_size = 100

def create_samples(data_mat):

    def draw100(data_mat):
        # randomly draw 100 ratings with replacement
        test_ind = np.random.randint(0, high=data_mat.shape[0], size=sample_size)
        test = data_mat[test_ind]
        
        # the remained ratings serve as prior set
        prior_ind = range(data_mat.shape[0])
        prior_ind = np.setdiff1d(prior_ind, test_ind)
        prior = data_mat[prior_ind]
        return test, prior
    
    tests = []
    for i in range(num_samples):
        test, prior = draw100(data_mat)
        filename = 'prior{}.txt'.format(i)
        with open(filename, 'wb') as f:
            np.savetxt(f, prior.astype(int), fmt='%i', delimiter='\t')
        tests.append(test)
    
    tests = np.vstack(tests)
    filename2 = 'tests.txt'
    with open(filename2, 'wb') as f2:
        np.savetxt(f2, tests.astype(int), fmt='%i', delimiter='\t')

def main():
    datafile = 'ml-100k/u.data'
    u_data = np.loadtxt(datafile, dtype=int)
    u_data = u_data[:, :-1]
    create_samples(u_data)
    
main()