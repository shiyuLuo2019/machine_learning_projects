import pickle
import sklearn
import numpy as np
from mnist import load_mnist
from sklearn import svm # this is an example of using SVM
import matplotlib.pyplot as plt
import shutil
import os

def preprocess(images):
    return np.array([i.flatten() for i in images])

def build_classifier(images, labels, C, gamma):
    classifier = svm.SVC(C=C, kernel='rbf', gamma=gamma)
    classifier.fit(images, labels)
    return classifier

##the functions below are required
def save_classifier(classifier, training_set, training_labels):
    #this saves the classifier to a file "classifier" that we will
    #load from. It also saves the data that the classifier was trained on.
    pickle.dump(classifier, open('classifier_1.p', 'w'))
    pickle.dump(training_set, open('training_set.p', 'w'))
    pickle.dump(training_labels, open('training_labels.p', 'w'))


def classify(images, classifier):
    #runs the classifier on a set of images. 
    return classifier.predict(images)

def error_measure(predicted, actual):
    return np.count_nonzero(abs(predicted - actual))/float(len(predicted))

def make_training_and_testing_set(flat_images, labels, n_training, n_testing):
    '''
    Construct training set, validation set and testing set.
    Parameters:
        - flat_images: numpy array, shape(n_images, n_features), n_features = n_columns * n_rows. floats.
        - labels: labels of images in <flat_images>, shape(n_images,), int.
        - n_training: number of training samples. int.
        - n_testing: number of testing samples. int.
    Returns:
        - training_set: numpy array, shape(n_training * n_classes, n_features), floats.
        - testing_set: numpy array, shape(n_testing * n_classes, n_features), floats.
        
    '''
    n_features = flat_images.shape[1]
    digits = np.unique(labels)
    
    training_set = np.zeros((1, n_features))
    testing_set = np.zeros((1, n_features))
   
    # construct labels
    training_labels = np.zeros((n_training * len(digits),))
    testing_labels = np.zeros((n_testing * len(digits),))
   
    
    for digit in digits:
        # find all images that belong to class 'digit'
        img_of_digit = flat_images[np.where(labels==digit)]
        
        # get training samples
        training_samples = img_of_digit[:n_training]
      
        # get testing samples
        testing_samples = img_of_digit[-n_testing:]
        
        # add to training set
        training_set = np.concatenate((training_set, training_samples), axis=0)
        training_labels[digit * n_training : (digit + 1) * n_training] = digit
        
        # add to testing set
        testing_set = np.concatenate((testing_set, testing_samples), axis=0)
        testing_labels[digit * n_testing : (digit + 1) * n_testing] = digit
        
    # remove the 0-th row
    training_set = training_set[1:]
    testing_set = testing_set[1:]
    
    return training_set, testing_set, training_labels, testing_labels


def confusion_matrix(actual_labels, predicted_labels):
    n_labels = len(np.unique(actual_labels))
    confusion_mat = np.zeros((n_labels, n_labels))
    for i in range(n_labels):
        predicted_labels_i = predicted_labels[np.where(actual_labels==i)]
        uniques, unique_counts = np.unique(predicted_labels_i, return_counts=True)
        confusion_mat[i, uniques.astype(int)] = unique_counts
    return confusion_mat

def visualize_misclassified(testing_set, actual_labels, predicted_labels):
    if os.path.isdir('svm_misclassified'):
        shutil.rmtree('svm_misclassified')
    os.mkdir('svm_misclassified')

    ind = np.nonzero(actual_labels - predicted_labels)
    flat_imgs = testing_set[ind]
    actuals = actual_labels[ind]
    predicteds = predicted_labels[ind]

    for flat_img, actual, predicted in zip(flat_imgs, actuals, predicteds):
        mat_img = np.reshape(flat_img, (28, 28))
        plt.imshow(mat_img,cmap='Greys_r')
        plt.axis('off')
        plt.savefig('svm_misclassified/{}_misclassified_as_{}.png'.format(int(actual), int(predicted)))                                                    

if __name__ == "__main__":

    # Code for loading data
    images, labels = load_mnist(digits=range(10), path='.')
    
    # preprocessing
    flat_images = preprocess(images)
    
    # pick training and testing set
    n_training = 3000 # first 3000 images of each class, 30000 in total
    n_testing = 100 # last 100 images of each class, 1000 in total
    training_set, testing_set, training_labels, testing_labels = make_training_and_testing_set(flat_images,labels, n_training, n_testing)
  
    
    # fit svm with data
    classifier = build_classifier(images=training_set, labels=training_labels, C=10., gamma=10./784.)
    
    # save classifier parameters
    save_classifier(classifier, training_set, training_labels)
    
    predicted = classify(testing_set, classifier)
    error_rate = error_measure(predicted, testing_labels)
    print 'error rate = {}'.format(error_rate)
   
