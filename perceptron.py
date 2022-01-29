import numpy as np
# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)

def stepFunction(t):
    if t >= 0:
        return 1    # y-hat = 1, data point is in the positive zone
    return 0        # y-hat = 0, data point is in the negative zone

def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])  

# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptronStep(X, y, W, b, learn_rate = 0.01):
    # Fill in code
    for i in range(len(X)):     #Going through every points in data.csv given in the exercise
        y_hat= prediction(X[i], W, b)   #predict for each point in which zone we should place it
        #We are only interested in checking misclassified points to be able to correct them
        if y[i]- y_hat == 1: #Misclassified positive points in negative zone. ( label positive(1) is in negative zone(0) ==> 1-0 = 0 )
            for j in range(len(W)):
                W[j] += (X[i][j]*learn_rate) #Fix each weight of for the point X[i][j]
            b += learn_rate #There's only 1 bias for each point X[i][j] so it is out of for j loop
        if y[i] - y_hat == -1: #Misclassified negative points in positive zone. ( label negative(0) is in positive zone(1) ==> 0-1 = -1)
            for j in range(len(W)):
                W[j] -= (X[i][j]*learn_rate) 
            b -= learn_rate
    return W, b
    
# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines
