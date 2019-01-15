import numpy as np
from preprocess import *
import time

class Decision(object):

    def __init__(self):
        pass
    
    # First,Generate a data set of size 20 by the procedure above and run the one-dimensional decisionstump algorithm on the data set. 
    # Record Ein. Repeat the experiment (including data generation, running the decision stump algorithm, and computing Ein) 5,000 times. 
    # What is the average Ein ?
    def calculate_Ein(self, X, Y):
        # calculate median of interval & negative infinite & positive infinite
        thetas = np.array([float("-inf")] + [(X[i] + X[i + 1]) / 2 for i in range(0, X.shape[0] - 1)] + [float("inf")])
        Ein = X.shape[0]
        sign = 1
        target_theta = 0.0
        
        ### YOUR CODE HERE
       
        for s in [-1.0, 1.0]:
            for theta in range(thetas.shape[0]):
                pred = s*np.sign(X-theta)
                error = np.sum(pred != Y)
                if error < Ein:
                    Ein = error
                    sign = s
                    target_theta = theta
        ### END YOUR CODE

        # two corner cases
        if target_theta == float("inf"):
            target_theta = 1.0
        if target_theta == float("-inf"):
            target_theta = -1.0
        Ein = Ein / X.shape[0] # mean of Ein
        return Ein, target_theta, sign
    
    # Repeat the experiment (including data generation, running the decision stump algorithm, and computing Ein) 5,000 times. 
    # What is the average Ein
    def decision_ray(self):
        T = 5000 # iteration
        Ein_all = [] # list for all Ein

        ### YOUR CODE HERE
        for i in range(T):
            np.random.seed(i)
            X, Y = generate_input_data(int(np.abs(time.time()/100*np.random.randn())))
            Ein, target_theta, sign = self.calculate_Ein(X, Y)
            print("iter[{}]---Ein:{}".format(i, Ein))
            Ein_all.append(Ein)
        ### END YOUR CODE

        # mean of Ein
        Ein_mean = np.mean(Ein_all)
        return Ein_mean
    
    # Run the algorithm on the Dtrain . What is the Ein of the optimal decision stump? 
    def decision_dtrain(self, path):
        X, Y = read_input_data(path)
        # record optimal descision stump parameters
        Ein = 1.0
        theta = 0
        sign = 1
        index = 0
        
        ### YOUR CODE HERE
        N = X.shape[0]
        M = X.shape[1]
        for i in range(M):
            error, target_theta, sign_ = self.calculate_Ein(X[:,i], Y)
            if error < Ein:
                Ein = error
                theta = target_theta
                sign = sign_
                index = i

        ### END YOUR CODE

        return Ein, theta, sign, index
    
    # Use the returned decision stump to predict the label of each example within the Dtest . Report an estimate Etest.
    def decision_dtest(self, path, theta, sign, index):
        
        ### YOUR CODE HERE
        X, Y = read_input_data(path)
        X = X[:, index]
        pred = sign*np.sign(X-theta)
        error = np.sum(pred != Y)
        Etest = error / Y.shape[0]
        ### END YOUR CODE
        
        return Etest