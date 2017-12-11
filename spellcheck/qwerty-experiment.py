#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This file contains experiments on qwerty_levenshtein_distance.

@author: sherryluo
"""
import random 
import time
import csv
import re
import numpy as np
from spellcheck import qwerty_levenshtein_distance
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def main(sample_size, num_trials):
    
    log_file = open('log/qwerty.txt','w')
    log_file.close()
    
    
    ## read dictionary
    dictionarywords = []
    with open('3esl.txt', 'r') as dict_file:
        dict_reader = csv.reader(dict_file)
        for row in dict_reader:
            dictionarywords.extend(row)
    
    ## read typos and truewords 
    typos = []
    truewords = []
    with open('wikipediatypo.txt', 'r') as f:
        for line in f:
            a_line = line.split() # read this line as a string and split by whitespaces
            typos_inline = a_line[0].split(',') # a list of typos in this line
            typos.extend(typos_inline)
            trues_inline = a_line[1].split(',') # a list of truewords in this line
            trueword = trues_inline[0] # picks the first trueword
            for i in range(len(typos_inline)):
                truewords.append(trueword)
               
    p = (1, 3, 5, 7)
    l = []
    for deletion_cost in p:
        for insertion_cost in p:
                error_rate = qwerty_experiment(deletion_cost, insertion_cost, 
                                               sample_size, num_trials, 
                                               typos, truewords, 
                                               dictionarywords)
                l.append([deletion_cost, insertion_cost, error_rate])
                
    ## plot and display figure
    ## the plotted figure will be saved under directory 'figures'
    print '\nfigure will be displayed 5 second and then saved\n'          
    l = np.array(l)
    plot_3d(l)
    
    
    
def plot_3d(l):
    xs = l[:, 0]
    ys = l[:, 1]
    dz = l[:, -1]
    zs = np.zeros(len(dz))
    dx = [0.1 for i in range(len(xs))]
    dy = [0.1 for i in range(len(xs))]
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    
    for x, y, z in zip(xs, ys, dz):
        text = format(z, '.3f')
        ax1.text(x, y, z, text, zdir=None)
    
    ax1.bar3d(xs, ys, zs, dx, dy, dz, color='#00ceaa')
    
    ax1.set_xlabel('deletion cost')
    ax1.set_ylabel('insertion cost')
    ax1.set_zlabel('average error rate')
    ax1.set_title('qwerty distance')
    
    plt.ion()
    plt.show()
    plt.pause(1)
    fig.savefig('figures/qwerty.png')
    
        
    
def qwerty_experiment(deletion_cost, insertion_cost, 
               sample_size, num_trials, typos, truewords, dictionarywords):
    
    """ An experiment is a an experiment on a certain combination
    of insertion cost and deletion cost. 
    It consists of *num_trials* many trials on such a combination,
    prints the *sample_size* and the time needed to run each trial,
    returns the mean error rate of these trials(i.e., this experiment),
    and prints the error rate of *typos* estimated by the mean error rate of
    these trials/this experiment. """
    
    print 'combination: deletion cost =', deletion_cost, 
    print ', insertion cost =', insertion_cost
    with open('log/qwerty.txt', 'a') as log_file:
        log_file.write('combination: deletion cost = ' + str(deletion_cost))
        log_file.write(', insertion cost = ' + str(insertion_cost))
  
    # initializes mean error rate to 0
    mean_error_rate = 0
    
    for i in range(0, num_trials):
        print '-' * 10, 'trial: ', i + 1, '-' * 10
        print 'sample size:', sample_size
        with open('log/qwerty.txt', 'a') as log_file:
            log_file.write('-' * 10 + 'trial: ' + str(i + 1) + '-' * 10 + '\n')
            log_file.write('sample size: ' + str(sample_size) + '\n')
            log_file.close()
        
        mean_error_rate += qwerty_trial(typos, truewords, dictionarywords,
                                 sample_size, 
                                 deletion_cost, 
                                 insertion_cost) / float(num_trials)
        
    print '\nestimated error rate:', mean_error_rate, '\n\n'
    with open('log/qwerty.txt', 'a') as log_file:
        log_file.write('-' * 10 + 'estimated error rate:'+ str(mean_error_rate) + '-' * 10 +'\n\n\n')
    return mean_error_rate


def qwerty_trial(typos, truewords, dictionarywords, sample_size, 
          deletion_cost, insertion_cost):
    """ Randomly chooses *sample_size* many samples from typos,
    prints the total time to correct this many typos by qwerty_levenshtein_distance algorithm,
    Returns the error rate of this trial based on the specified *deletion_cost*
    and *insertion_cost*. """
    
    ## randomly choose *sample_size* many typos
    ## by generating random numbers
    typos_subset = []
    truewords_subset = []
    for i in range(sample_size):
        index = random.randrange(0, len(typos))
        typos_subset.append(typos[index])
        truewords_subset.append(truewords[index])
    
    ## prints the total time to correct *sample_size* many words,
    ## and returns the error rate of this trial
    error_rate = qwerty_measure_time_error_rate(typos_subset, 
                                         truewords_subset, 
                                         dictionarywords,
                                         deletion_cost, insertion_cost)
    
    
    return error_rate

def qwerty_measure_time_error_rate(typos, truewords, dictionarywords, 
                                   deletion_cost, insertion_cost):
    """ Runs qwerty_measure_error(args...) on all words in *typos* 
    and corresponding *truewords*, prints the time consumed 
    , and returns the error rate. """
   
    ## how long does it take to run measure_error on the sampled data?
    start = time.time()
    error_rate = qwerty_measure_error(typos, truewords, dictionarywords,
                                      deletion_cost, insertion_cost)
    end = time.time()
    
    print 'total time consumed:', end - start
    with open('log/qwerty.txt', 'a') as log_file:
        log_file.write('total time consumed:' + str(end - start) + '\n')
    
    return error_rate


def qwerty_measure_error(typos, truewords, dictionarywords, 
                         deletion_cost, insertion_cost):
    """ Given *deletion_cost* and *insertion_cost*, 
    returns the corresponding error rate as a real value between 0 and 1,
    based on qwerty-levenshtein-distance algorithm. """
    
    num_errors = 0
    
    for i in range(0, len(typos)):
        closest = qwerty_find_closest_word(typos[i], dictionarywords, 
                                           deletion_cost, insertion_cost)
        if closest != truewords[i]:
            num_errors += 1
            
    return float(num_errors) / len(typos)


def qwerty_find_closest_word(typo, dictionary, deletion_cost, insertion_cost):
    """ Given the *deletion_cost* and *insertion_cost*, 
    finds the closest word to *typo* in *dictionary*
    based on qwerty_levenshtein_distance algorithm in spellcheck.py. """

    curr_closest = dictionary[0]
    curr_distance = qwerty_levenshtein_distance(all_alphanumeric(typo), all_alphanumeric(dictionary[0]), 
                                         deletion_cost, insertion_cost)

    for each in dictionary:
        distance = qwerty_levenshtein_distance(all_alphanumeric(typo), all_alphanumeric(each), 
                                               deletion_cost, insertion_cost)
        if distance < curr_distance:
            curr_closest = each
            curr_distance = distance

    return curr_closest

def all_alphanumeric(word):
    """ return a new string with non-alphanumeric characters in *word* removed.
    The original *word* is not modified. """
    return re.sub(r'[^0-9a-zA-Z]+', '', word)
    
main(30, 3)