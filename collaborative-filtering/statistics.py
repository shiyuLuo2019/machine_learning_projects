#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 19:38:49 2017

@author: sherryluo
"""
import csv
import numpy as np
from scipy import stats
from math import ceil
import matplotlib.pyplot as plt

U = 943 # number of users
I = 1682 # number of movies

def reviews(freq_table, f):
    """
    Finds the movies that have the most or fewest reviews based on 
    *freq_table* and *f*.
    Attributes:
        *freq_table*: a I-by-2 matrix, where the first column specifies the 
            movie IDs, the second column shows the numbers of reviews of 
            each movie.
        *f* is either:
            - np.amax (finds movies with most reviews)
            - np.amin (finds movies with fewest reviews)
    """
    
    movies = freq_table[:, 0]
    freqs = freq_table[:, 1]
    extreme_freq = f(freqs) # the largest/smallest frequency
    extreme_movies = movies[np.where(freqs == extreme_freq)] # the movies that receive most/fewest reviews
    return (extreme_movies, extreme_freq)   

def sort_movies(freq_table):
    """
    Sort the movies in ascending order according to their number of reviews.
    Attribute:
        *freq_table*: a I-by-2 matrix, where the first column specifies the 
            movie IDs, the second column shows the numbers of reviews of 
            each movie.
    """
    
    return freq_table[freq_table[:, 1].argsort()]


def user_review_list(userids, itemids):
    """
    Finds all movies that are reviewed by every single user.
    """
    
    # L[0] = an empty list
    # L[u] = movies rated by user u
    L = []
    L.append([])
    zipped = np.array(zip(userids, itemids))
    zipped = zipped[zipped[:, 0].argsort()]
    
    # finds movies rated by every single user
    for u in range(1, U + 1):
        # finds movies rated by user u
        movies = zipped[zipped[:, 0] == u]
        movies = movies[:, 1]
        L.append(np.copy(movies))
    return L


def plot_hist(num_reviewed, num_userpairs):
    length = np.amax(num_reviewed)
    interval = 14
    num_groups = int(ceil(float(length) / interval))
    
    normalized_num_userpairs = []
    for i in range(num_groups - 1):
        normalized_num_userpairs.append(sum(num_userpairs[i * interval : (i + 1) * interval - 1]))
    normalized_num_userpairs.append(sum(num_userpairs[interval * num_groups - 1:]))
    
    fig, ax = plt.subplots()
    
    index = 2.5 * np.arange(num_groups)
    bar_width = 2.5
    
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    
    rects = plt.bar(index, normalized_num_userpairs, bar_width, alpha=opacity, color='r',
                    error_kw=error_config)
    
    plt.xlabel('number of movies that a pair of users both rated')
    plt.ylabel('number of user pairs that have rated that many movies')
    l_boundaries = [i * interval for i in range(num_groups)]
    plt.xticks(index - bar_width / 2, l_boundaries)
    #r_boundaries = [(i + 1) * interval - 1 for i in range(num_groups -1)]
    #r_boundaries.append(length)
    #plt.xticks(index + bar_width, r_boundaries)
    ax.tick_params(axis='x', labelsize=6)
   
    plt.savefig('prob1a.png')
    plt.ion()
    plt.pause(10)
    plt.show()
   

def plot_line(ranks, freqs):
    plt.plot(ranks, freqs, '-o')
    plt.xlabel('rank in frequency table')
    plt.ylabel('frequency')
    plt.savefig('prob1b.png')
    plt.show()
    plt.ion()
    plt.pause(10)
    plt.show()
    
def main():
    u_data = csv.reader(open('ml-100k/u.data', 'rb'), delimiter='\t')
    columns = list(zip(*u_data))
    # column 1: user id
    col1 = np.array(columns[0]).astype(np.int)
    # column 2: item id
    col2 = np.array(columns[1]).astype(np.int)    
    
    # review_list[u] = a list of movies that were reviewed by user u + 1
    review_list = user_review_list(col1, col2)[1:]
  
    mat = np.zeros(U * (U - 1))
    cnt = 0    
    for i in range(U):
        for j in range(U):
            if i != j:
                mat[cnt] = len(np.intersect1d(review_list[i], review_list[j]))
                cnt = cnt + 1     
    
    # on average, how many movies are commonly reviewed by a user pair?
    mean = np.mean(mat)
    # median number of movies commonly reviewed by a user pair
    median = np.median(mat)
    # how many user pairs have rated that many movies?
    freq_table1 = stats.itemfreq(mat)
    maximum = freq_table1[-1, 0]
    minimum = freq_table1[0, 0]
    
    # display results
    print 'mean:', mean
    print 'median:', median
    # print freq_table1[:, 0]
    interval = 10
    plot_hist(freq_table1[:, 0], freq_table1[:, 1])
    
    # measure how many reviews each movie has
    freq_table2 = stats.itemfreq(col2)
    # which movies have the most/fewest reviews?
    most = reviews(freq_table2, np.amax)
    fewest = reviews(freq_table2, np.amin)
    # display results
    print 'movies that have the most reviews:', most[0]
    print 'number of reviews:', most[1]
    print 'movies that have the fewest reviews:', fewest[0]
    print 'number of reviews:', fewest[1]
    
    # sort the movies based on their number of reviews
    sorted_movies = sort_movies(freq_table2)
    # display results
    plot_line(np.arange(len(sorted_movies[:, 0])), sorted_movies[:, 1])
    
    
main()
    
    
    
    
    
            
    




