#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: sherryluo
"""
import sys

class DataReader: # path: string
    """ This class represents a data reader.
    Attributes:
        path: path of the data file, represented as a string.
        training_set_size: size of training set
    """
    def __init__(self, path):
        """ Return a data reader object whose file is the
        one opened by the specified *path*.
        """
        self.file = open(path, 'r')
        self.examples = []
        self.attributes = ()

    def init_examples(self):
        """ Initializes the examples as a dictionary 
        of feature-vector:label pairs.
        A feature vectors are represented as a dictionary:
            E.X.: {'GoodGrade': true, 'IsRich': false, ...}
        """
        
        # returns the corresponding boolean-typed value of *str*
        def str2bool(str):
            return str == 'True'or str == 'true'
        
        # returns the corresponding list of boolean-typed value of *strlist*
        # e.x.: ['True', 'False', 'True'] -> [True, False, True]
        def strlist2boollist(strlist):
            boollist = []
            for str in strlist:
                boollist.append(str2bool(str))
            return boollist
    
        title = self.file.readline().split()
        self.attributes = tuple(title[0:len(title) - 1])
        
        # read each row into a tuple: (feature_val1, feature_val2, ..., label)
        # append each tuple into examples list
        for line in self.file:
            an_example = strlist2boollist(tuple(line.split()))
            self.examples.append(an_example)
           

    def get_examples(self):
        """ Return a copy of the original example list to ensure
        this data reader can be used multiple times without destroying
        the original data."""
        
        return list(self.examples)
    
    def get_attributes(self):
        """ Return a tuple of attributes """
        return self.attributes
    
        
    
        


     
        
    
