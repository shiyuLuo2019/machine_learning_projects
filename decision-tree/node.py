#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This module contains a class, which represents a node in a binary tree.
It has a value, a left child and a right child.

@author: sherryluo
"""

class Node:
    """ This class represents node in a binary decision tree.
    It has a val and its left and right child (node or label).
    """
    def __init__(self, val):
        self.val = val
        self.left = True
        self.right = True
        
    def set_left(self, left):
        self.left = left
    
    def set_right(self, right):
        self.right = right
    
    def get_val(self):
        return self.val
    
    def get_left(self):
        return self.left
    
    def get_right(self):
        return self.right
        
    
    
    
    

    
