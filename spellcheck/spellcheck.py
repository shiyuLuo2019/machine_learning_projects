#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 12:15:46 2017

@author: sherryluo
"""
import sys
import csv
import re
import time
import numpy as np


""" initializes the row-col-coordinate for each key on the keyboard. """
row_top = '1 2 3 4 5 6 7 8 9 0'.split()
row0_upper = 'Q W E R T Y U I O P'.split()
row0_lower = 'q w e r t y u i o p'.split()
row1_upper = 'A S D F G H J K L'.split()
row1_lower = 'a s d f g h j k l'.split()
row2_upper = 'Z X C V B N M'.split()
row2_lower = 'z x c v b n m'.split()
keys = [row0_upper, row0_lower, 
        row1_upper, row1_lower, 
        row2_upper, row2_lower]
## a dictionary mapping each key on keyboard to a row-col coordinate
## e.x.: 'Q':(1, 0), 'g':(2, 4), '0':(0, 9)
keyboard = {}
for i in range(len(row_top)):
    keyboard[row_top[i]] = (0, i)
for i in range(0, len(keys) / 2):
    for j in range(0, len(keys[i * 2])):
        keyboard[keys[i * 2][j]] = (i + 1, j)
        keyboard[keys[i * 2 + 1][j]] = (i + 1, j)
           

def main():
    
    dictionary = []
    with open(sys.argv[2], 'r') as dict_file:
        dict_reader = csv.reader(dict_file)
        for row in dict_reader:
            dictionary.extend(row)
    dict_file.close()
    
    with open(sys.argv[1], 'r') as query_file:
        input_queries = query_file.read()
    query_file.close()
    
    queries = re.sub(r'[^0-9a-zA-Z]+', '\n', input_queries)
    queries = queries.split()

    corrected = []
    for word in queries:
        corrected.append(find_closest_word(word, dictionary))
    
    dlmt = re.sub(r'[0-9a-zA-Z]+', 'a', input_queries)
    dlmt = dlmt.split('a')
    
    output = []
    for i in range(len(corrected)):
        output.append(dlmt[i])
        output.append(corrected[i])
    output.append(dlmt[len(dlmt) - 1])
    
    corrected_file = open('corrected.txt', 'w')
    for chunk in output:
        corrected_file.write(chunk)
        
        
def measure_error(typos, truewords, dictionarywords):
    num_errors = 0
    
    for i in range(0, len(typos)):
        closest = find_closest_word(typos[i], dictionarywords, 1, 1, 1)
        if closest != truewords[i]:
            num_errors += 1
            
    return float(num_errors) / len(typos)
    

def find_closest_word(string1, dictionary):
    """ Finds the closest string in *dictionary* to *string1*. 
    *dictionary* is a list of strings. """
  
    curr_closest = dictionary[0]
    curr_distance = levenshtein_distance(string1, dictionary[0], 1, 1, 1)
    for each in dictionary:
        distance = levenshtein_distance(string1, each, 1, 1, 1)
        if distance < curr_distance:
            curr_closest = each
            curr_distance = distance
            
    return curr_closest
            


def qwerty_levenshtein_distance(s, t, deletion_cost, insertion_cost):
    
    def compute_substitution_cost(key1, key2):
        """ Computes the substitution cost according to *key1* and *key2*'s
        Manhattan distance on keyboard. """
    
        return abs(keyboard[key1][0] - keyboard[key2][0]) + abs(keyboard[key1][1] - keyboard[key2][1])
    
    m = len(s)
    n = len(t)
    
    cost_matrix = np.zeros((m + 1, n + 1), dtype=np.int)
    
    for i in range(0, m + 1):
        cost_matrix[m, 0] = i * deletion_cost
        
    for j in range(n + 1, 0):
        cost_matrix[0, n] = j * insertion_cost

    for i in range(1, n + 1):
        cost_matrix[0, i] = i * insertion_cost
        
    for j in range(1, m + 1):
        cost_matrix[j, 0] = j * deletion_cost
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost_matrix[i, j] = min([cost_matrix[i - 1, j] + deletion_cost,
                       cost_matrix[i, j - 1] + insertion_cost,
                       cost_matrix[i - 1, j - 1] + compute_substitution_cost(s[i - 1],
                                  t[j - 1])])
    return cost_matrix[m, n]
        

def levenshtein_distance(s, t, 
                        deletion_cost, insertion_cost, substitution_cost):
    """ for all i and j, 
    cost_matrix[i, j] will hold the levinshtein distance
    between the first i characters of *s* 
    and the first j characters of *t*
    """
    
    def compute_substitution_cost(char1, char2):
        if char1 == char2:
            return 0
        else:
            return substitution_cost
    
    m = len(s)
    n = len(t)
    
    cost_matrix = np.zeros((m + 1, n + 1), dtype=np.int)
    
    for i in range(0, m + 1):
        cost_matrix[m, 0] = i * deletion_cost
        
    for j in range(n + 1, 0):
        cost_matrix[0, n] = j * insertion_cost
      
    for i in range(1, n + 1):
        cost_matrix[0, i] = i * insertion_cost
        
    for j in range(1, m + 1):
        cost_matrix[j, 0] = j * deletion_cost
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost_matrix[i, j] = min([cost_matrix[i - 1, j] + deletion_cost,
                       cost_matrix[i, j - 1] + insertion_cost,
                       cost_matrix[i - 1, j - 1] + compute_substitution_cost(s[i - 1],
                                  t[j - 1])])
    return cost_matrix[m, n]

main()       
    
                
