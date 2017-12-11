import sys
import csv
import numpy as np
import scipy

MAX_ITERS = 10

def perceptrona(w_init, X, Y):
	#figure out (w, k) and return them here. w is the vector of weights, k is how many iterations it took to converge.
    k = 0 # number of iterations
    w = w_init #initializes w
    
    m = np.shape(Y)[0] # number of training examples
    x = np.array([np.ones(m), X]).T # adds an extra column to X
    y = Y # 1d array
    
    def sign(num):
        if num > 0:
            return +1
        else :
            return -1
    
    def all_classified(x, y, w):
        products = x.dot(w)
        for j in range(np.shape(products)[0]):
            if sign(products[j]) * y[j] < 0:
                return False
        
        return True
        
    def misclassified(xi, yi, w):
        dot_product = w.T.dot(xi)
        prediction = sign(dot_product[0])
        return prediction * yi < 0
        
        
    i = 0 # x[i] = i-th training example, y[i] = label of i-th training example
    while True:
        if k > MAX_ITERS: 
            print 'exceeds maximal iterations'
            return (w, float('inf'))
            
        if i == 0:
            if all_classified(x, y, w):
                print 'w:', w
                print 'number iterations:', k
                return (w, k)
            else:
                k = k + 1
                
        
        if misclassified(x[i], y[i], w):
            w = w + y[i] * np.array([x[i]]).T
            
        i = (i + 1) % m

def main():
    
    rfile = sys.argv[1]
    	
    #read in csv file into np.arrays X1, X2, Y1, Y2
    csvfile = open(rfile, 'rb')
    dat = csv.reader(csvfile, delimiter=',')
    X1 = []
    Y1 = []
    X2 = []
    Y2 = []
    for i, row in enumerate(dat):
        if i > 0:
            X1.append(float(row[0]))
            X2.append(float(row[1]))
            Y1.append(float(row[2]))
            Y2.append(float(row[3]))
            
    X1 = np.array(X1)
    X2 = np.array(X2)
    Y1 = np.array(Y1)
    Y2 = np.array(Y2)
    
    m = np.shape(Y1)[0]
    w_init = np.array([[0], [0]])
    perceptrona(w_init, X1, Y1)
    # uncomment the following line to run sequential perceptron on X2-Y2
    # perceptrona(w_init, X2, Y2)

if __name__ == "__main__":
	main()
