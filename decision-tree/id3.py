#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This module contains functions for building a binary decision tree.

@author: sherryluo
"""

from math import log
from node import Node

def entropy(s):
    """ Computes the entropy of set *s*"""
    
    def count():
        cnt = 0
        for i in range(0, len(s)):
            if (s[i][-1]):
                cnt = cnt + 1
        return cnt
    
    num = len(s)
    num_true = count()
    if (num_true == 0 or num_true == num):
        return 0
    else:
        p_true = (num_true) / float(num)
        p_false = 1 - p_true
        return -1 * (p_true * log(p_true, 2) + p_false * log(p_false, 2))


def info_gain(s, n):
    
    """ Computes the information gain of the *n*-th attribute
    based on the given set *s*. """
    
    branch1 = []
    branch2 = []
    
    for i in range(0, len(s)):
        an_example = s[i]
        if (an_example[n]):
            branch1.append(an_example)
        else:
            branch2.append(an_example)

    p1 = len(branch1) / float(len(s))  # = P(value of attribute n = true | s)
    p2 = len(branch2) / float(len(s))  # = P(value attribute n = false | s)
    gain = entropy(s) - p1 * entropy(branch1) - p2 * entropy(branch2)
    return gain

def choose_attribute(attribute_indices, s):
    """ Choose the best attribute (its index) in *attribute_indices* 
    that maximize information gain given set *s*.
    attribute_indices: a list of integers representing the i-th attribute,
    e.x.: for IvyLeague.txt, [0, 1, 3] = 'GoodGrades', 'GoodLetters', 'IsRich'
    """
    max_gain = info_gain(s, attribute_indices[0])
    best = attribute_indices[0]

    for i in range(1, len(attribute_indices)):
        gain = info_gain(s, attribute_indices[i])
        if (gain > max_gain): 
            max_gain = gain
            best = attribute_indices[i]
    return best


def DTL(examples, attribute_indices, default):
    """ Returns a decision tree using ID3 algorithm,
    given training data *examples* """
    def mode(expls):
        count_true = 0
        for i in range(0, len(expls)):
            if (expls[i][-1]):
                count_true = count_true + 1
        if (count_true > (len(expls) - count_true)):
            return True
        else:
            return False

    def remove(attribute_indices, a):
        """ Returns attribute_indices - a. """
        copy = list(attribute_indices)
        copy.remove(a)
        return copy
    
    def same_labels():
        """ Are all examples in the training set classified 
        as the same class (labels)? """
        first = examples[0][-1]
        for i in range(0, len(examples)):
            if (examples[i][-1] != first):
                return False
        return True
    
    def split(atb_index, value, training_set):
        """ Extracts all examples in *training_set* that are classified
        as *value* and return a list containing these examples.
        The original list *training_set* will not be affected.
        """
        result = []
        for i in range(0, len(training_set)):
            if (training_set[i][atb_index] == value):
                result.append(training_set[i])
        return result
    
    if (len(examples) == 0):
        return default
    elif (same_labels()):
        return examples[0][-1]
    elif (len(attribute_indices) == 0):
        return mode(examples)
    else:
        best = choose_attribute(attribute_indices, examples)
        tree = Node(best)

        examples_true = split(best, True, examples)
        left = DTL(examples_true, 
                   remove(attribute_indices, best), 
                   mode(examples_true))
        tree.set_left(left)

        examples_false = split(best, False, examples)
        right = DTL(examples_false, 
                    remove(attribute_indices, best), 
                    mode(examples_false))
        tree.set_right(right)
        
    return tree

def classify(input_tree, feature_vector):
    """ Computes the estimated output given the input tree and
    feature vector. """
    if (type(input_tree) is bool):
        return input_tree
    
    if (feature_vector[input_tree.get_val()]):
        return classify(input_tree.get_left(), feature_vector)
    else:
        return classify(input_tree.get_right(), feature_vector)


  