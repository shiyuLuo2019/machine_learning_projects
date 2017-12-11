#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 16:39:36 2017

@author: Shiyu Luo
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from scipy.spatial.distance import cityblock

def which_metric(distance):
    if distance == 1:
        return cityblock
    else:
        def pearson_v2(u, v):
            return pearsonr(u, v)[0]
        return pearson_v2

def knn(mat, target_row, nonzero_col, metric, k, iFlag):
    
    def nearest(matrix, target_vector, m):
        metric = which_metric(m)
        distance_matrix = cdist(matrix, np.array([target_vector]), metric)
        if m == 1: #manhattan
            index = np.argmin(distance_matrix)
        else: #pearson
            index = np.argmax(distance_matrix)
        return index, matrix[index]
        
    # original mat stay intact
    mat_copy = np.copy(mat)
    target_vector = mat[target_row]
    mat_copy = np.delete(mat_copy, target_row, axis=0)

    if iFlag == 0:
        # filter out all users whose rating for *movieid* is 0
        mat_copy = mat_copy[mat_copy[:, nonzero_col] > 0]
        
    if mat_copy.shape[0] <= k:
        return mat_copy
    
    # num_neighbors = mat_copy.shape[0]
    neighbors = np.zeros((k, mat.shape[1]))
    for i in range(k):
        # find k nearest user neighbors
        index, neigh = nearest(mat_copy, target_vector, metric)
        # write that nearest user vector into neighbors
        neighbors[i, :] = neigh
        # delete that user vector from mat_copy
        mat_copy = np.delete(mat_copy, (index), axis=0)
    return neighbors


