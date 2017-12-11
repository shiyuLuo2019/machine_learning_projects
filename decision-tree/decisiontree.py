#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This file contains a main() function that
can process binary decision tree problems.
"""

# read input options
# input format: inputFileName, trainingSetSize, numberOfTrials, verbose
import sys
import random
from datareader2 import DataReader2
import id3
import utils

def main():
    if (len(sys.argv) != 5):
        sys.exit("invalid command-line arguments format")

    # handling command-line arguments
    data = DataReader2(sys.argv[1])
    data.init_examples()
    training_set_size = int(sys.argv[2])
    num_trials = int(sys.argv[3])
    verbose = int(sys.argv[4])
    if (verbose != 1 and verbose != 0):
        sys.exit("invalid command-line argument")
    if (num_trials < 1):
        sys.exit("invalid command-line argument")
    
    # extract examples and attributes
    # an example = a feature vector + a label (represented by a tuple)
    examples = data.get_examples() # a list of examples
    attributes = data.get_attributes()  # a list of attribute names
    if (training_set_size >= len(examples)):
        sys.exit("invalid command-line argument")

    # lists of classification performances (correct rates)
    # e.x.: [1.0, 0.95, 0.83, ...]
    correct_rates_id3 = []
    correct_rates_prior = []
    
    for i in range(0, num_trials): # a single trial
        print 'TRIAL NUMBER:', i + 1
        print '-' * 30
        
        # randomly pick a training set of size *training_set_size*
        random.shuffle(examples)
        training_examples = examples[0:training_set_size]
        testing_examples = examples[training_set_size:]
        
        # a list of actual labels of testing examples
        actuals = utils.extract_labels(testing_examples)
        
        # build a decision tree based on these training examples
        tree = id3.DTL(training_examples, range(0, len(attributes)), True)
        # print the structure of the decision tree built from the training set
        print 'DECISION TREE STRUCTURE'
        utils.print_tree(tree, attributes)
    
        # list of predicted labels using id3
        output_id3_1 = utils.trial_id3(tree, testing_examples)
        # list of predicted labels using prior probability
        output_prior_1 = utils.trial_priorprob(training_examples, testing_examples)
        
        # computes and prints correct rate of this trial
        correct_rate_id3 = utils.correct_rate(output_id3_1, actuals)
        correct_rates_id3.append(correct_rate_id3)
        correct_rate_prior = utils.correct_rate(output_prior_1, actuals)
        correct_rates_prior.append(correct_rate_prior)
        print '\n'
        print 'proportion of correct classification'
        print 'decision tree:', correct_rate_id3
        print 'prior probability:', correct_rate_prior
        print '\n'
        
        if (verbose == 1):
            output_id3_2 = list(testing_examples)
            output_prior_2 = list(testing_examples)
            for j in range(0, len(output_id3_2)):
                output_id3_2[j][-1] = output_id3_1[j]
                output_prior_2[j][-1] = output_prior_1[j]
                
            print '*' * 10, 'examples in the training set: ', '*' * 10
            utils.print_dataset(training_examples, attributes)
            
            print '*' * 10, 'examples in the testing set: ', '*' * 10
            utils.print_dataset(testing_examples, attributes)
            
            print '*' * 10, 'classification by the decision tree: ', '*' * 10
            utils.print_dataset(output_id3_2, attributes)
            
            print '*' * 10, 'classification by prior probability: ', '*' * 10
            utils.print_dataset(output_prior_2, attributes)
            
    # other outputs
    print '*' * 5, 'information', '*' * 5
    print 'file:' + sys.argv[1]
    print 'training set size:', sys.argv[2]
    print 'testing set size:', len(examples) - int(sys.argv[2])
    print 'number of trials:', num_trials
    mean_tree = utils.mean(correct_rates_id3)
    mean_prior = utils.mean(correct_rates_prior)
    print 'mean classification performance (decision tree):', mean_tree
    print 'mean classification performance (prior probability):', mean_prior


main()
