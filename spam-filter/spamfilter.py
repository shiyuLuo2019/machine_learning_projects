#Starter code for spam filter assignment in EECS349 Machine Learning
#Author: Prem Seetharaman (replace your name here)

import sys
import numpy as np
import os
import shutil
from math import log

def parse(text_file):
	#This function parses the text_file passed into it into a set of words. Right now it just splits up the file by blank spaces, and returns the set of unique strings used in the file. 
	content = text_file.read()
	return np.unique(content.split())

def writedictionary(dictionary, dictionary_filename):
	#Don't edit this function. It writes the dictionary to an output file.
	output = open(dictionary_filename, 'w')
	header = 'word\tP[word|spam]\tP[word|ham]\n'
	output.write(header)
	for k in dictionary:
		line = '{0}\t{1}\t{2}\n'.format(k, str(dictionary[k]['spam']), str(dictionary[k]['ham']))
		output.write(line)
		

def makedictionary(spam_directory, ham_directory, dictionary_filename):
    #Making the dictionary. 
    spams = [f for f in os.listdir(spam_directory) if os.path.isfile(os.path.join(spam_directory, f)) and f != '.DS_Store']
    hams = [f for f in os.listdir(ham_directory) if os.path.isfile(os.path.join(ham_directory, f)) and f != '.DS_Store']

    words = {}
    spam_prior_probability = len(spams)/float((len(spams) + len(hams)))

    words_in_spams = []
    words_in_hams = []
    num_spams = len(spams)
    num_hams = len(hams)
    for spam in spams:
        with open(spam_directory + '/' + spam, 'r') as f:
            words_in_spam = parse(f)
        words_in_spams.extend(words_in_spam)
    for ham in hams:
        with open(ham_directory + '/' + ham, 'r') as f:
            words_in_ham = parse(f)
        words_in_hams.extend(words_in_ham)
        
    # unique_spamwords: all unique words in all spam documents
    # occurrence_spamwords[i] = how many documents have word unique_spamwords[i]
    # prob_spamwords[i] = given a spam document, the probability the word
    # unique_spamwords[i] appears in this document
    s1 = 1.0 # spam pseudocount parameter
    s0 = 1.0 # spam pseudocount parameter
    unique_spamwords, occurrence_spamwords = np.unique(words_in_spams, return_counts=True)
    prob_spamwords = (occurrence_spamwords + s1) / float(num_spams + s1 + s0)
    
    h1 = 1.0 # ham pseudocount parameter
    h0 = 1.0 # ham pswudocount parameter                                       
    unique_hamwords, occurrence_hamwords = np.unique(words_in_hams, return_counts=True)
    prob_hamwords = (occurrence_hamwords + h1) / float(num_hams + h0 + h1)
    
    all_words = np.unique(np.concatenate((unique_spamwords, unique_hamwords), axis=0))
    for word in all_words:
        # probability that this word appears given a spam
        if word in unique_spamwords:
            ind = np.nonzero(unique_spamwords == word)[0][0]
            prob_inspam = prob_spamwords[ind]
        else:
            prob_inspam = s0 / float(num_spams + s1 + s0)
        # probability that this word appears given a ham
        if word in unique_hamwords:
            ind = np.nonzero(unique_hamwords == word)[0][0]
            prob_inham = prob_hamwords[ind]
        else:
            prob_inham = h0 / float(num_hams + h1 + h0)
        words[word] = {'spam':prob_inspam, 'ham':prob_inham}
        
    writedictionary(words, dictionary_filename)
    return words, spam_prior_probability


def is_spam(content, dictionary, spam_prior_probability):
    prob_spam = log(spam_prior_probability)
    prob_ham = log(1.0 - spam_prior_probability)
    # extracts all words from the dictionary
    dict_words = dictionary.keys()

    for word in dict_words:
        if word in content:
            prob_spam += np.log(dictionary[word]['spam'])
            prob_ham += np.log(dictionary[word]['ham'])
        else:
            prob_spam += np.log(1.0 - dictionary[word]['spam'])
            prob_ham += np.log(1.0 - dictionary[word]['ham'])

    return prob_spam > prob_ham

def spamsort(mail_directory, spam_directory, ham_directory, dictionary, spam_prior_probability):
    mail = [f for f in os.listdir(mail_directory) if os.path.isfile(os.path.join(mail_directory, f)) and f != '.DS_Store']
    counter = 1
    for m in mail:
        content = parse(open(mail_directory + '/' + m))
        spam = is_spam(content, dictionary, spam_prior_probability)
        if spam:
            shutil.copy(mail_directory + '/' + m, spam_directory)
        else:
            shutil.copy(mail_directory + '/' + m, ham_directory)
            
            
if __name__ == "__main__":
	#Here you can test your functions. Pass it a training_spam_directory, a training_ham_directory, and a mail_directory that is filled with unsorted mail on the command line. It will create two directories in the directory where this file exists: sorted_spam, and sorted_ham. The files will show up  in this directories according to the algorithm you developed.
    training_spam_directory = sys.argv[1]
    training_ham_directory = sys.argv[2]
    
    test_mail_directory = sys.argv[3]
    test_spam_directory = 'sorted_spam'
    test_ham_directory = 'sorted_ham'
	
    if not os.path.exists(test_spam_directory):
        os.mkdir(test_spam_directory)
    if not os.path.exists(test_ham_directory):
        os.mkdir(test_ham_directory)
    
    dictionary_filename = "dictionary.dict"
    
    
    #create the dictionary to be used
    dictionary, spam_prior_probability = makedictionary(training_spam_directory, training_ham_directory, dictionary_filename)
    #sort the mail
    spamsort(test_mail_directory, test_spam_directory, test_ham_directory, dictionary, spam_prior_probability) 
