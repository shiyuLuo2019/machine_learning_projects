#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from itertools import izip
import copy
import math


def main():
    """
    This function runs your code for problem 2.

    You can also use this to test your code for problem 1,
    but make sure that you do not leave anything in here that will interfere
    with problem 2. Especially make sure that gmm_est does not output anything
    extraneous, as problem 2 has a very specific expected output.
    """
    file_path = sys.argv[1]
    
    # prepare data
    X1, X2 = read_gmm_file(file_path)
    n_samples1 = len(X1)
    n_samples2 = len(X2)
    X1 = np.reshape(X1, (1, n_samples1))
    X2 = np.reshape(X2, (1, n_samples2))
   
    # fit class 1 to GMM
    mu_init1 = np.array([[10.], [30.]])
    sigmasq_init1 = np.array([[25.], [15.]])
    wt_init1 = np.array([.7, .3])
    its = 30
    mu_results1, sigma2_results1, w_results1, L1 = gmm_est(X1, mu_init1, sigmasq_init1, wt_init1, its)
    
    # fit class2 to GMM
    mu_init2 = np.array([[-25.], [-5.], [50.]])
    sigmasq_init2 = np.array([[5.], [5.], [100.]])
    wt_init2 = np.array([.3, .4, .3])
    mu_results2, sigma2_results2, w_results2, L2 = gmm_est(X2, mu_init2, sigmasq_init2, wt_init2, its)
    
    # format parameters
    mu_results1 = np.reshape(mu_results1, (2,))
    sigma2_results1 = np.reshape(sigma2_results1, (2,))
    mu_results2 = np.reshape(mu_results2, (3,))
    sigma2_results2 = np.reshape(sigma2_results2, (3,))

    # mu_results1, sigma2_results1, w_results1 are all numpy arrays
    # with learned parameters from Class 1
    print 'Class 1'
    print 'mu =', mu_results1, '\nsigma^2 =', sigma2_results1, '\nw =', w_results1

    # mu_results2, sigma2_results2, w_results2 are all numpy arrays
    # with learned parameters from Class 2
    print '\nClass 2'
    print 'mu =', mu_results2, '\nsigma^2 =', sigma2_results2, '\nw =', w_results2
    
    # plot log-likelihoods before EM step and at each of the first 20 iterations
    plot_loglikelihood_histories(L1[:21], L2[:21])

def gmm_est(X, mu_init, sigmasq_init, wt_init, its):
    """
    Input Parameters:
      - X             : user input data, numpy array, shape(n_features, n_samples), floats
      - mu_init       : initial means, numpy array, shape(n_features, K), floats
      - sigmasq_init  : initial covariances, numpy array, shape(K, n_features, n_features), floats
      - wt_init       : initial weights, numpy array, shape(K, ), floats, sums to 1
      - its           : number of iterations for the EM algorithm

    Returns:
      - mu            : mean of each mixture component, numpy array, shape(n_features, K)
      - sigmasq       : covariance of each mixture component, numpy array, shape(K, n_features, n_features)
      - wt            : weight of each mixture component, numpy array, shape(K, ), sums to 1
      - L             : Log-likelihood of this gaussian mixture
    """
    
    wt = copy.deepcopy(wt_init)
    mu = copy.deepcopy(mu_init)
    sigmasq = copy.deepcopy(sigmasq_init)
    L = np.zeros((its+1,))
    L[0] = compute_likelihood(X=X.T, wt=wt, mu=mu, cov=sigmasq)
    for i in range(its):
        resp = compute_responsibilities(X.T, wt, mu, sigmasq)
        wt = estimate_wt(resp)
        mu = estimate_mean(X.T, resp)
        sigmasq = estimate_covariance(X.T, resp, wt, mu)
        L[i+1] = compute_likelihood(X=X.T, wt=wt, mu=mu, cov=sigmasq)
    return mu, sigmasq, wt, L
    

def pdf(mean, cov):
    '''
    Returns the gaussian probability density function.
    Parameters:
        - mean: mean of the distribution, array-like, shape(n_variables, ), floats.
        - cov: covariance matrix of the distribution, array-like, shape(n_variable, n_variable), floats.
    '''
    probability_density_function = lambda x : multivariate_normal.pdf(x, mean=mean, cov=cov)
    return probability_density_function


def compute_responsibilities(X, wt, mu, cov):
    '''
    Computes the responsibilities of each mixture component for x_n.
    Parameters:
        - X: user input data, numpy array, shape(n_samples, n_features), floats
        - wt: weights of each mixture component, shape(K,), floats, sums to 1.
        - mu: means of each mixture component, numpy array, shape(K, n_features), floats.
        - cov: covariance matrices of each mixture component, numpy array, shape(K, n_features, n_features), floats.
    Returns:
        - responsibilities: responsibilities of each mixture component for X, numpy array, shape(n_samples, K), floats.
    '''

    conditional_probs = [np.apply_along_axis(pdf(mu_k, cov_k), arr=X, axis=1) for mu_k, cov_k in izip(mu, cov)]
    conditional_probs = np.array(conditional_probs).T
    joint_probs = conditional_probs * wt
    independent_probs = np.sum(joint_probs, axis=1)
    posterior_probs = (joint_probs.T / independent_probs).T
    return posterior_probs

def estimate_wt(responsibilities):
    '''
    Estimates the weights of each mixture component. 
    Parameters:
        - responsibilities: a responsibility matrix, numpy array, shape(n_samples, K), floats.
    Returns:
        new_wt: updated weights of each mixture component, numpy array, shape(K,), floats, sums to 1
    '''
    new_wt = np.mean(responsibilities, axis=0)
    return new_wt


def estimate_mean(X, responsibilities):
    '''
    Parameters:
        - X: user input data, numpy array, shape(n_samples, n_features), floats
        - responsibilities: a responsibility matrix, numpy array, shape(n_samples, K), floats
    Returns:
        new_mean: updated means of each mixture component, numpy array, shape(K, n_features), floats
    '''
    n_samples = np.shape(X)[0]
    divisor = estimate_wt(responsibilities) * n_samples
    new_mean = (np.matmul(X.T, responsibilities) / divisor).T
    return new_mean
    
def estimate_covariance(X, responsibilities, wt, mu):
    '''
    Parameters:
        - X: user input data, numpy array, shape(n_samples, n_features), floats
        - responsibilities: a responsibility matrix, numpy array, shape(n_samples, K), floats
        - wt: weights of each mixture components
        - mu: means of each mixture components
    '''
    K= len(wt)
    n_samples, n_features = np.shape(X)
    covariance_matrices = np.empty((K, n_features, n_features))
    for k in range(K):
        # normalize X
        X_norm = X - mu[k]
        covariance_matrices[k] = np.dot(responsibilities[:, k] * X_norm.T, X_norm) / (wt[k] * n_samples)
    return covariance_matrices
    

def compute_likelihood(X, wt, mu, cov):
    '''
    Parameters:
        - X: user input data, numpy array, shape(n_samples, n_features), floats
        - mu: means of each mixture component, shape(K, n_features), floats
        - cov: covariance matrices of each mixture component, numpy array, shape(K, n_features, n_features), floats
        - wt: weights of each mixture component, numpy array, shape(K, ), floats, sums to 1
    Returns:
        log_likelihood: Log-likelihood that all samples in X appear given the mixture model.
    '''
    
    individual_probability = lambda x_n : compute_individual_probability(x_n, wt, mu, cov)
    probabilities = np.apply_along_axis(func1d=individual_probability, axis=1, arr=X)
    log_likelihood = np.sum(np.log(probabilities))   
    return log_likelihood


def compute_individual_probability(x_n, wt, mu, cov):
    '''
    Computes the probability that x_n would appear given the gaussian mixture model.
    Parameters:
        - x_n: a sample, numpy array, shape(n_features,), floats.
        - wt: weights of each mixture components, shape(K,), floats, sums to 1
        - mu: means of each mixture components, shape(K, n_features), floats
        - cov: covariance matrices of each mixture components, shape(K, n_features, n_features), floats
    Returns:
        - probability: the probability would appear given the mixture model, floats
    '''
    probability = np.sum([wt_k * multivariate_normal.pdf(x_n, mean=mu_k, cov=cov_k) for wt_k, mu_k, cov_k in izip(wt, mu, cov)])
    # print probability
    return probability

def plot_loglikelihood_histories(likelihood_history1, likelihood_history2):
    fig1 = plt.figure()
    iters = np.arange(21, dtype='int')
    plt.plot(iters, likelihood_history1, label='class 1')
    plt.xlabel('iteration')
    plt.ylabel('log-likelihood')
    plt.legend()
    plt.xticks(iters)
    plt.title('log-likelihood at each iteration')
    plt.savefig('likelihood_class1.png')
    
    fig2 = plt.figure()
    plt.plot(iters, likelihood_history2, label='class 2')
    plt.xlabel('iteration')
    plt.ylabel('log-likelihood')
    plt.legend()
    plt.xticks(iters)
    plt.title('log-likelihood at each iteration')
    plt.savefig('likelihood_class2.png')
    
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
