#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import sys
import matplotlib.pyplot as plt
from gmm_est import gmm_est
from math import log
from scipy.stats import multivariate_normal
from itertools import izip


def main():
    """
    This function runs your code for problem 3.

    You can use this code for problem 4, but make sure you do not
    interfere with what you need to do for problem 3.
    """
    file_path = sys.argv[1]
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    X = data[:, :-1]
    Y = data[:, -1]
    n_samples = len(Y)
    X = np.reshape(X, (1, n_samples))
    
    # estimated parameters from fitting gmm_train.csv
    mu1 = np.array([[9.77488592], [29.58258718]])
    sigmasq1 = np.array([[[21.92280456]], [[9.78376961]]])
    wt1 = np.array([0.59765463, 0.402345371])
    mu2 = np.array([[-24.82275173],  [-5.06015828], [49.62444472]])
    sigmasq2 = np.array([[[7.94733541]], [[23.32266181]], [[100.0243375]]])
    wt2 = np.array([0.20364946, 0.49884302, 0.29750752])
    p1 = 1000./3000.
    class_pred = gmm_classify(X, mu1, sigmasq1, wt1, mu2, sigmasq2, wt2,p1)
    
    class1_data = X[:, np.where(class_pred==1)]
    class1_data = class1_data[0, 0, :]
    # class1_data is a numpy array containing
    # all of the data points that your gmm classifier
    # predicted will be in class 1.
    print 'Class 1'
    print class1_data
    
    class2_data = X[:, np.where(class_pred==2)]
    class2_data = class2_data[0, 0, :]
    # class2_data is a numpy array containing
    # all of the data points that your gmm classifier
    # predicted will be in class 2.
    print '\nClass 2'
    print class2_data
    
    # compute accuracy
    n_correct = np.sum(Y==class_pred)
    accuracy = float(n_correct) / n_samples
    
    # plot histogram
    X1 = X[0, np.where(Y==1)]
    X1 = X1[0]
    X2 = X[0, np.where(Y==2)]
    X2 = X2[0]
    plot_hist(X1, X2, class1_data, class2_data, accuracy)
    
    
def plot_hist(X1, X2, class1_data, class2_data, accuracy):
    '''
    Plots the histogram for X1 and X2, and marks the class predicted for each data point.
    Parameters:
        - X1: data of class1, numpy array, shape(n_samples,), floats
        - X2: data of class2, numpy array, shape(n_samples,), floats
        - class1_data: predicted data of class 1, numpy array, shape(n_samples,), floats
        - class2_data: predicted data of class 2, numpy array, shape(n_samples,), floats
    '''

    def add_rand_jitter(arr, deviation):
        '''
        Addes random jitter to arr.
        '''
        n_points = len(arr)
        rand_jitter = np.random.randn(n_points)
        return deviation + arr + rand_jitter
    
    n_bins = 50
    colors = ['red', 'blue']
    labels = ['class 1', 'class 2']
    
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.set_xlabel('input value')
    ax.set_ylabel('frequency')
    ax.set_title('Two-color histogram of gmm_test.csv, accuracy={}'.format(accuracy))
    ax.hist([X1, X2], n_bins, color=colors, alpha=.6, label=labels)
    
    y1 = add_rand_jitter(np.ones((len(class1_data),)), 5.)
    y2 = add_rand_jitter(np.ones((len(class2_data),)), 0.)
    ax.scatter(class1_data, y1, marker='.', s=1, color='m', label='predicted as class 1')
    ax.scatter(class2_data, y2, marker='.', s=1, color='c', label='predicted as class 2')
    ax.legend(prop={'size': 10})
    plt.savefig('two-color-hist-w-predicted-class.png')
    
    
def gmm_classify(X, mu1, sigmasq1, wt1, mu2, sigmasq2, wt2, p1):
    """
    Input Parameters:
        - X           : input data, numpy array, shape(n_features, n_samples), floats
        - mu1: means of each mixture component of class 1, numpy array, shape(K1, n_features), floats
        - cov1: covariance matrices of each mixture component of class 1, numpy array, shape(K1, n_features, n_features), floats
        - wt1: weight of each mixture component of class 1, numpy array, shape(K1,), floats, sums to 1
        - mu2: means of each mixture component of class 2, numpy array, shape(K2, n_features), floats
        - cov2: covariance matrices of each mixture component of class 2, numpy array, shape(K2, n_features, n_features), floats
        - wt2: weight of each mixture component of class 2, numpy array, shape(K2,), floats, sums to 1
        - p1: prior probability of class 1

    Returns:
        - class_pred  : a numpy array containing results from the gmm classifier
                        (the results array should be in the same order as the input data points)
    """
    classify = lambda x: classify_individual_sample(x, mu1, sigmasq1, wt1, mu2, sigmasq2, wt2, p1)
    class_pred = np.apply_along_axis(func1d=classify, axis=0, arr=X)
    return class_pred


def classify_individual_sample(x, mu1, cov1, wt1, mu2, cov2, wt2, p1):
    '''
    Classify x.
    Parameters:
        - x: an input sample, numpy array, shape(n_features,), floats.
        - mu1: means of each mixture component of class 1, numpy array, shape(K1, n_features), floats
        - cov1: covariance matrices of each mixture component of class 1, numpy array, shape(K1, n_features, n_features), floats
        - wt1: weight of each mixture component of class 1, numpy array, shape(K1,), floats, sums to 1
        - mu2: means of each mixture component of class 2, numpy array, shape(K2, n_features), floats
        - cov2: covariance matrices of each mixture component of class 2, numpy array, shape(K2, n_features, n_features), floats
        - wt2: weight of each mixture component of class 2, numpy array, shape(K2,), floats, sums to 1
        - p1: prior probability of class 1
    Returns:
        predicted label of x(1 or 2)
    '''
    log_posterior_prob1 = log(compute_conditional_prob(x, mu1, cov1, wt1)) + log(p1)
    log_posterior_prob2 = log(compute_conditional_prob(x, mu2, cov2, wt2)) + log(1. - p1)
    if log_posterior_prob1 > log_posterior_prob2:
        return 1
    else:
        return 2


def compute_conditional_prob(x, mu, cov, wt):
    '''
    Computes the probability that x appears given the class.
    Parameters:
        - x: an input sample, numpy array, shape(n_features,), floats.
        - mu: means of each mixture component of the conditioned class, numpy array, shape(K, n_features), floats
        - cov: covariance matrices of each mixture component of the conditioned class, numpy array, shape(K, n_features, n_features), floats
        - wt: weights of each mixture component of the conditioned class, numpy array, shape(K,), floats, sums to 1
    '''
    conditional_prob = np.sum([multivariate_normal.pdf(x, mu_k, cov_k) for mu_k, cov_k in izip(mu, cov)])
    return conditional_prob
    

def read_gmm_file(path_to_file):
    """
    Reads either gmm_test.csv or gmm_train.csv
    :param path_to_file: path to .csv file
    :return: two numpy arrays for data with label 1 (X1) and data with label 2 (X2)
    """
    X1 = []
    X2 = []

    data = open(path_to_file).readlines()[1:] # we don't need the first line
    for d in data:
        d = d.split(',')

        # We know the data is either class 1 or class 2
        if int(d[1]) == 1:
            X1.append(float(d[0]))
        else:
            X2.append(float(d[0]))

    X1 = np.array(X1)
    X2 = np.array(X2)

    return X1, X2

if __name__ == '__main__':
    main()
