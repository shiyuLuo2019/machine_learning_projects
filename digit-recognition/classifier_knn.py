import pickle
import sklearn
from sklearn.neighbors import KNeighborsClassifier # this is an example of using SVM
from classifier_1 import make_training_and_testing_set
import sys
from mnist import load_mnist
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

def preprocess(images):
    # resize the images so that 
    return np.array([i.flatten() for i in images])

def build_classifier(images, labels, k):
    classifier = KNeighborsClassifier(n_neighbors=k, weights='distance')
    classifier.fit(images, labels)
    return classifier

##the functions below are required
def save_classifier(classifier, training_set, training_labels):
    #this saves the classifier to a file "classifier" that we will
    #load from. It also saves the data that the classifier was trained on.
    pickle.dump(classifier, open('classifier_2.p', 'w'))
    pickle.dump(training_set, open('training_set_2.p', 'w'))
    pickle.dump(training_labels, open('training_labels_2.p', 'w'))


def classify(images, classifier):
    #runs the classifier on a set of images. 
    return classifier.predict(images)

def error_measure(predicted, actual):
    return np.count_nonzero(abs(predicted - actual))/float(len(predicted))

def confusion_matrix(actual_labels, predicted_labels):
    n_labels = len(np.unique(actual_labels))
    confusion_mat = np.zeros((n_labels, n_labels))
    for i in range(n_labels):
        predicted_labels_i = predicted_labels[np.where(actual_labels==i)]
        uniques, unique_counts = np.unique(predicted_labels_i, return_counts=True)
        confusion_mat[i, uniques.astype(int)] = unique_counts
    return confusion_mat

def visualize_misclassified(testing_set, actual_labels, predicted_labels):
    if os.path.isdir('knn_misclassified'):
        shutil.rmtree('knn_misclassified')
    os.mkdir('knn_misclassified')

    ind = np.nonzero(actual_labels - predicted_labels)
    flat_imgs = testing_set[ind]
    actuals = actual_labels[ind]
    predicteds = predicted_labels[ind]

    for flat_img, actual, predicted in zip(flat_imgs, actuals, predicteds):
        mat_img = np.reshape(flat_img, (28, 28))
        plt.imshow(mat_img,cmap='Greys_r')
        plt.axis('off')
        plt.savefig('knn_misclassified/knn_{}_misclassified_as_{}.png'.format(int(actual), int(predicted))) 

if __name__ == "__main__":
    
    n_training = 3000
    n_testing = 100
    K = 4
   
    # Code for loading data
    images, labels = load_mnist(digits=range(10), path='.')
    
    # preprocessing
    flat_images = preprocess(images)
    
    # pick training and testing set
    training_set, testing_set, training_labels, testing_labels = make_training_and_testing_set(flat_images,labels, n_training, n_testing)
    
    # fit svm with data
    classifier = build_classifier(images=training_set, labels=training_labels, k=K)
   
    # save classifier parameters
    save_classifier(classifier, training_set, training_labels)

    classifier = pickle.load(open('classifier_2.p'))
    
    predicted_labels = classify(testing_set, classifier)
    print error_measure(predicted_labels, testing_labels)
