#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This file contains experiments on levenshtein_distance.

@author: sherryluo
"""
import sys
import random 
import time
import csv
import numpy as np
from spellcheck import levenshtein_distance
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def main(sample_size, num_trials):
    deletion_cost = int(sys.argv[1])
    
    ## log file
    log_file = open('log/delcost-' + str(deletion_cost) + '.txt', 'w')
    log_file.close()
    
    ## read dictionary
    dictionarywords = []
    with open('3esl.txt', 'r') as dict_file:
        dict_reader = csv.reader(dict_file)
        for row in dict_reader:
            dictionarywords.extend(row)
    dict_file.close()
    
    ## read typos and truewords 
    typos = []
    truewords = []
    with open('wikipediatypo.txt', 'r') as f:
        f_reader = csv.reader(f)
        for row in f_reader:
            pair = row[0].split()
            typos.append(pair[0])
            truewords.append(pair[1])
    f.close()
   
    p = (0, 1, 2, 4)
    l = []
    for insertion_cost in p:
        for substitution_cost in p:
            error_rate = experiment(deletion_cost, insertion_cost, substitution_cost, sample_size, num_trials, typos, truewords, dictionarywords)
            l.append([insertion_cost, substitution_cost, error_rate])
    
    ## plot and display figure
    ## the plotted figure will be saved under directory 'figures'
    print '\nfigure will be displayed 5 second and then saved\n'
    plot(deletion_cost, l)
    
  
def plot(deletion_cost, l):
    """ Plots the average error rate of all possible combinations
    of insertion cost and substitution cost, 
    given the fixed deletion cost. """
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    
    num_elements = len(l)
    # starting points
    xs = []
    ys = []
    zs = [0 for i in range(num_elements)]
    
    ## changes
    dx = [0.1 for i in range(num_elements)]
    dy = [0.1 for i in range(num_elements)]
    dz = []
    
    for i in range(0, len(l)):
        xs.append(l[i][0])
        ys.append(l[i][1])
        dz.append(l[i][2])
    
    for x, y, z in zip(xs, ys, dz):
        text = format(z, '.3f')
        ax1.text(x, y, z, text, zdir=None)
    
    ax1.bar3d(xs, ys, zs, dx, dy, dz, color='#00ceaa')
              
    ax1.set_title('deletion cost =' + str(deletion_cost))
    ax1.set_xlabel('insertion cost')
    ax1.set_ylabel('substitution cost')
    ax1.set_zlabel('error rate')
    
    plt.ion()
    plt.show()
    plt.pause(30)
    fig.savefig('figures/del_cost-' + str(deletion_cost) + '.png')


def experiment(deletion_cost, insertion_cost, substitution_cost,
               sample_size, num_trials, typos, truewords, dictionarywords):
    """ An experiment is a an experiment on a certain combination
    of insertion cost, deletion cost, and substitution cost. 
    It consists of *num_trials* many trials on a certain combination,
    prints the *sample_size* and the time needed to run each trial,
    returns the mean error rate of these trials(i.e., this experiment),
    and prints the error rate of *typos* estimated by the mean error rate of
    these trials/this experiment. """
    
    print 'combination: deletion cost =', deletion_cost, 
    print ', insertion cost =', insertion_cost,
    print ', substitution cost =', substitution_cost
    with open('log/delcost-' + str(deletion_cost) + '.txt', 'a') as log_file:
        log_file.write('combination: deletion cost = ' + str(deletion_cost))
        log_file.write(', insertion cost = ' + str(insertion_cost))
        log_file.write(', substitution cost = ' + str(substitution_cost) + '\n')
    
    # initializes mean error rate to 0
    mean_error_rate = 0
    
    for i in range(num_trials):
        print '-' * 10, 'trial: ', i + 1, '-' * 10
        print 'sample size:', sample_size
        with open('log/delcost-' + str(deletion_cost) + '.txt', 'a') as log_file:
            log_file.write('-' * 10 + 'trial: ' + str(i + 1) + '-' * 10 + '\n')
            log_file.write('sample size: ' + str(sample_size) + '\n')
            log_file.close()
        
        mean_error_rate += trial(typos, truewords, dictionarywords,
                                 sample_size, deletion_cost, insertion_cost,
                                 substitution_cost) / float(num_trials)
    
    print '\nestimated error rate:', mean_error_rate, '\n\n'
    with open('log/delcost-' + str(deletion_cost) + '.txt', 'a') as log_file:
        log_file.write('\nestimated error rate:'+ str(mean_error_rate) + '\n\n\n')
    
    return mean_error_rate


def trial(typos, truewords, dictionarywords, sample_size, 
          deletion_cost, insertion_cost, substitution_cost):
    """ Randomly chooses *sample_size* many samples from typos,
    Prints the total time to correct this many typos.
    Returns the error rate of this trial, given *insertion_cost*,
    *deletion_cost*, and *substitution_cost*. """
    
    ## randomly choose *sample_size* many typos
    ## by generating random numbers
    typos_subset = []
    truewords_subset = []
    for i in range(0, sample_size):
        index = random.randrange(0, len(typos))
        typos_subset.append(typos[index])
        truewords_subset.append(truewords[index])
    
    ## prints the total time to correct *sample_size* many words,
    ## and returns the error rate of this trial
    error_rate = measure_time_error_rate(typos_subset, 
                                         truewords_subset, 
                                         dictionarywords, 
                                         deletion_cost, insertion_cost, 
                                         substitution_cost)
    
    
    return error_rate


def measure_time_error_rate(typos, truewords, dictionarywords, 
                            deletion_cost, insertion_cost, substitution_cost):
    """ Runs spellcheck.measure_error_aux(args...) on typos 
    and corresponding truewords, prints the time consumed 
    , and returns the error rate.  """
   
    ## how long does it take to run measure_error on the sampled data?
    start = time.time()
    error_rate = measure_error(typos, truewords, dictionarywords, 
                               deletion_cost, insertion_cost, substitution_cost)
    end = time.time()
    
    print 'total time consumed:', end - start
    with open('log/delcost-' + str(deletion_cost) + '.txt', 'a') as log_file:
        log_file.write('total time consumed:' + str(end - start) + '\n')
    
    return error_rate


def measure_error(typos, truewords, dictionarywords, deletion_cost, insertion_cost, substitution_cost):
    """ Returns the error rate as a real value between 0 and 1,
    given the deletion cost, insertion cost and substitution cost. """
    
    num_errors = 0
    
    for i in range(0, len(typos)):
        closest = find_closest_word(typos[i], dictionarywords, 
                                    deletion_cost, insertion_cost, 
                                    substitution_cost)
        if closest != truewords[i]:
            num_errors += 1
            
    return float(num_errors) / len(typos)


def find_closest_word(string1, dictionary, deletion_cost, insertion_cost, substitution_cost):
    """ Finds the closest string in *dictionary* to *string1*,
    given the deletion cost, insertion cost, and substitution cost.
    *dictionary* is a list of strings. """
  
    curr_closest = dictionary[0]
    curr_distance = levenshtein_distance(string1, dictionary[0], 
                                            deletion_cost, 
                                            insertion_cost, 
                                            substitution_cost)
    for each in dictionary:
        distance = levenshtein_distance(string1, each, 
                                           deletion_cost, 
                                           insertion_cost, 
                                           substitution_cost)
        if distance < curr_distance:
            curr_closest = each
            curr_distance = distance
            
    return curr_closest
          
main(30, 3)
