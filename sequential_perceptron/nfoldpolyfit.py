import sys
import csv
import math
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.colors as cls

def nfoldpolyfit(X, Y, maxK, n, verbose):
#	NFOLDPOLYFIT Fit polynomial of the best degree to data.
#   NFOLDPOLYFIT(X,Y,maxDegree, nFold, verbose) finds and returns the coefficients 
#   of a polynomial P(X) of a degree between 1 and N that fits the data Y 
#   best in a least-squares sense, averaged over nFold trials of cross validation.
#
#   P is a vector (in numpy) of length N+1 containing the polynomial coefficients in
#   descending powers, P(1)*X^N + P(2)*X^(N-1) +...+ P(N)*X + P(N+1). use
#   numpy.polyval(P,Z) for some vector of input Z to see the output.
#
#   X and Y are vectors of datapoints specifying  input (X) and output (Y)
#   of the function to be learned. Class support for inputs X,Y: 
#   float, double, single
#
#   maxDegree is the highest degree polynomial to be tried. For example, if
#   maxDegree = 3, then polynomials of degree 0, 1, 2, 3 would be tried.
#
#   nFold sets the number of folds in nfold cross validation when finding
#   the best polynomial. Data is split into n parts and the polynomial is run n
#   times for each degree: testing on 1/n data points and training on the
#   rest.
#
#   verbose, if set to 1 shows mean squared error as a function of the 
#   degrees of the polynomial on one plot, and displays the fit of the best
#   polynomial to the data in a second plot.
#   
#
#   AUTHOR: Shiyu Luo
    
    columnized_X = np.array([X]).T # columnize X
    columnized_Y = np.array([Y]).T # columnize Y
    M = len(X) # how many examples are there in X?     
    
    avg_mse = [] # records average MSE. avg_mse[1] = the average MSE when maximum degree = 1
    for k in range(maxK + 1):  # try degree = 0, 1, 2, ..., maxK
        # construct design matrix based on columnized_X and degree
        design_matrix = make_design_matrix(columnized_X, k)
    
        # computes the average MSE given the design_matrix with n-fold cv
        mse = nfoldcv(n, design_matrix, columnized_Y)
        avg_mse.append(mse)

    avg_mse = np.array(avg_mse) # convert to a numpy array
    
    # computes coefficients
    best_degree = np.argmin(avg_mse)
    design_matrix_best = make_design_matrix(columnized_X, best_degree)
    best_w = regress(design_matrix_best, columnized_Y)
    
    if verbose == 1:
        plot(avg_mse, 'degree-MSE.png')
        
        xs = np.linspace(-1, 1, 500)
        ys = predict(np.array([xs]).T, best_degree, best_w)
        plot2(xs, ys, X, Y, 'best-fit.png')
    
    return best_w


def plot(ys, name):
    """ Plots the points (index, ys[index]) for every element in *ys*, 
    and save the figure as *name*.
    Attributes:
        *ys*: a numpy array representing y values. The index of each element
              will be used as its corresponding x value
        *name*: the path to save the plotted figure
    """

    xs = range(len(ys))
    alphas = [1 - y / float(max(ys)) for y in ys]
    colors = [cls.to_rgba((1.0, 0, 0, alpha)) for alpha in alphas]
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 1, 1)
    plt.xlim(-1, len(ys) + 2)
    plt.xticks(xs)
    plt.scatter(xs, ys, color='red')
    
    plt.ion()
    plt.show()
    plt.pause(3)
    plt.savefig(name)
   
    
def plot2(xs, ys, Xs, Ys, name):
    """
    Plots each (x, y) pair in xs and ys, and each (X, Y) pair in Xs and Ys.
    Saves the plotted figure as *name*.
    Attributes:
        *xs*: a numpy array
        *ys*: a numpy array
        *X*: a numpy array 
        *Y*: a numpy array
    Constraints:
        len(ys) == xs, len(Ys) == Xs
    """
    
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 1, 1)
    plt.plot(xs, ys, color='blue', linewidth=2, linestyle='-', label='regression')
    plt.legend(loc='upper left')
    plt.xlim(-1.0, 1.0)
    
    plt.scatter(Xs, Ys, color='red')
    plt.ion()
    plt.show()
    plt.pause(3)
    plt.savefig(name)
    


def predict(X, degree, w):
    """
    Predicts the outcome given a column vector of input values, maximal degree
    of polynomial regression, and coefficients vector
    Attributes:
        *X*: a column vector
        *degree*: maximal degree
        *w*: a column vector of coefficients
    Returns:
        a numpy array of predicted values
    """

    design_matrix = make_design_matrix(X, degree)
    y = np.dot(design_matrix, w)
    return y[:, 0]   
    
    
def make_design_matrix(X, k):
    """
    Constructs the design matrix given the column vector *X* and
    maximal degree *k*.
    Attributes:
        *X*: a column vector of feature values
        *k*: maximal degree
    Return:
        a design_matrix
    """
    
    M = len(X) # number of examples
    
    # construct design matrix
    design_matrix = np.ones(M)
    design_matrix = np.array([design_matrix]).T
    i = 0
    while (i < k):
        # element-wise multiplication
        new_col = np.multiply(np.array([design_matrix[:, -1]]).T, X)
        # insert a new column to the design matrix
        design_matrix = np.concatenate((design_matrix, new_col), axis=1)
        i += 1
    return design_matrix
    
    
def nfoldcv(n, design_matrix, Y):
    """ 
    Conducts n-fold cross validation on *design_matrix* and outcome vector *Y*.
    Attributes:
        *n*: n-fold
        *design_matrix*: a design_matrix containing feature vectors 
                        of all examples
                        *Y*: a column vector containing all outcomes
    Return: 
        average of mean square errors
    """
    
    ## split into n folds
    M = len(Y) # number of examples in design_matrix
    fold_size = M / n
    remain = M % n
    
    folds = []
    mse_history = []
    upperbound = 0
    for i in range(n):
        lowerbound = upperbound + fold_size
        if remain > 0:
            lowerbound = lowerbound + 1
            remain = remain - 1
            
        if i == 0:
            testing_fold = (design_matrix )
            
        testing_fold_X = design_matrix[upperbound:lowerbound, :]
        testing_fold_Y = Y[upperbound:lowerbound]
        
        if i == 0:
            training_folds_X = design_matrix[lowerbound:, :]
            training_folds_Y = Y[lowerbound:]
        else:
            training_folds_X_1 = design_matrix[:upperbound, :]
            training_folds_X_2 = design_matrix[lowerbound:, :]
            training_folds_X = np.concatenate((training_folds_X_1, training_folds_X_2), axis=0)
            training_folds_Y_1 = Y[:upperbound, :]
            training_folds_Y_2 = Y[lowerbound:, :]
            training_folds_Y = np.concatenate((training_folds_Y_1, training_folds_Y_2), axis=0)
        
        w = regress(training_folds_X, training_folds_Y) # computes coefficients
        if w is not None: 
            mse = mean_square_error(testing_fold_X, testing_fold_Y, w) # computes mse
            mse_history.append(mse)
            upperbound = lowerbound # update upperbound
        
        # since features are independent,
        # must be invertible
        
    average_mse = sum(mse_history) / n
    return average_mse


def regress(X, Y):
    """ Returns the coeffieicnt vector of X. 
    If X * tranpose(X) is not invertible, a column vector of 1's will be returned.
    Attributes:
        *X*: design_matrix of feature vectors. 
        X[i][j] = the j-th feature of the i-th example
        *Y*: a column vector of outcomes. Y[i] = the outcome of i-th example.
    """
    
    try:
        inv = np.linalg.inv(np.dot(X.T, X))
    except np.linalg.LinAlgError:
        # not invertible. return None
        return None
    else:
        return np.dot(np.dot(inv, X.T), Y)
    

def mean_square_error(X, Y, w):
    """ Computes the mean square error.
    Attributes:
        *X*: a matrix of feature vectors
        *Y*: a column vector of actual outcomes
        *w*: a column vector of coefficients 
    Returns:
        the mean square error
    """
    predicts = np.dot(X, w)
    square_error = sum(np.square(predicts - Y))[0]
    return float(square_error) / len(Y)


def main():
	# read in system arguments, first the csv file, max degree fit, number of folds, verbose
	rfile = sys.argv[1]
	maxK = int(sys.argv[2])
	nFolds = int(sys.argv[3])
	verbose = int(sys.argv[4])
	
	csvfile = open(rfile, 'rb')
	dat = csv.reader(csvfile, delimiter=',')
	X = []
	Y = []
	# put the x coordinates in the list X, the y coordinates in the list Y
	for i, row in enumerate(dat):
		if i > 0:
			X.append(float(row[0]))
			Y.append(float(row[1]))
	X = np.array(X)
	Y = np.array(Y)
	nfoldpolyfit(X, Y, maxK, nFolds, verbose)

if __name__ == "__main__":
	main()
    
    
