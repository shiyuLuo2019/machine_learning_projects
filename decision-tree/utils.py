#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This module contains data processing helper functions.

@author: sherryluo
"""

import id3
import node

def correct_rate(results, actuals):
    """ Computes the proportion of correct output. """ 
    
    num_correct = 0
    for i in range(0, len(results)):
        if (results[i] == actuals[i]):
            num_correct = num_correct + 1
    return float(num_correct) / len(actuals)
        
def mean(list_rates):
    """ Computes the mean of the given list of rates. """
    
    return sum(list_rates) / len(list_rates)

def print_dataset(dataset, attributes):
    def print_example(example):
        for i in range(0, len(example) - 1):
            print attributes[i],
            print ":",
            print example[i],
            print '\t',
        print 'Class',
        print ':',
        print example[-1], '\n'  
    for i in range(0, len(dataset)):
        print_example(dataset[i])
    
    
def trial_id3(tree, testing_examples):
    """ A single trial on *testing_examples* using *tree*. """
    
    list_classes = []
    for i in range(0, len(testing_examples)):
        feature_vector = testing_examples[i][:-1]
        list_classes.append(id3.classify(tree, feature_vector))
    
    return list_classes


def trial_priorprob(training_examples, testing_examples):
    """ A single trial on *testing_examples* by applying
    the prior probability algorithm. """
    
    def most_likely():
        """ What is the mostly likely class in *training_examples*? 
        If # True == # False, return False """
        
        cnt_true = 0
        for i in range(0, len(training_examples)):
            if (training_examples[i][-1]):
                cnt_true = cnt_true + 1
        cnt_false = len(training_examples) - cnt_true
        if (cnt_true > cnt_false):
            return True
        else:
            return False
        
    most_likely_class = most_likely()
    return [most_likely_class] * len(testing_examples)
    

def extract_labels(dataset):
    """ Extracts labels (classes) in all the examples of
    *dataset* and return a list containing these labels (classes).
    The original *dataset* will not be affected.
    """
    
    labels = []
    for i in range(0, len(dataset)):
        labels.append(dataset[i][-1])
    return labels

def print_tree(tree, attributes):
    """ Prints the given decision tree *tree*. 
    Since values of nodes in tree are indices of corresponding
    attributes, *attributes* must be a list containing the 'name'
    of these attributes.
    """
    def print_tree_aux(subtree, parent):
        # in case where subtree itself is boolean
        # e.x.: a training set whose all examples are classified
        # as True.
        if (type(subtree) is bool):
            print subtree
            return
            
        leftEnded = False
        rightEnded = False 

        print 'parent:', parent, 'attribute:', attributes[subtree.get_val()],
        if (type(subtree.get_left()) is bool):
            print 'trueChild:', subtree.get_left(),
            leftEnded = True # left subtree has ended
        else:
            print 'trueChild:', attributes[subtree.get_left().get_val()],
            
        if (type(subtree.get_right()) is bool):
            print 'falseChild:', subtree.get_right()
            rightEnded = True # right subtree has ended
        else:
            print 'falseChild:', attributes[subtree.get_right().get_val()]
        
        if (not leftEnded):
            print_tree_aux(subtree.get_left(), attributes[subtree.get_val()])
        if (not rightEnded):
            print_tree_aux(subtree.get_right(), attributes[subtree.get_val()])
            
    print_tree_aux(tree, 'root')
        
    
    