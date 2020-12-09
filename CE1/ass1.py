import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("example_data.csv",delimiter=',')
data_x, data_y = np.hsplit(data,2)

#scatter = plt.scatter(data_x, data_y,s=15)
#plt.show()

def cost(x, y, theta):
    # Compute cost for linear regression
    # J is the cost using theta as the parameter for linear regression to fit the data points in X and y

    ###################################################################################
    #                                   YOUR CODE HERE                                #
    ###################################################################################
    J=0
    
    x_1 = np.ones([np.shape(x)[0],1])
    x_1 = np.hstack([x_1,x])

    for i in range(np.shape(x)[0]):
        J = J+0.5*((sum(x_1[i]*theta)-y[i])**2)
    ###################################################################################
    #                                  END OF YOUR CODE                               #
    ###################################################################################

    return J

# Implement Gradient descent algorithm
def gradient_descent(x, y, theta, alpha, num_iters):
    # gradient_descent performs gradient descent to learn theta
    # gradient_descent updates theta by taking num_iters gradient steps with learning rate alpha
    
    for iter in range(num_iters):

        ###################################################################################
        #                                   YOUR CODE HERE                                #
        ###################################################################################
        theta = theta +  alpha * np.dot(np.squeeze(y - (theta[1]*data_x+theta[0])),np.hstack([np.ones([np.shape(x)[0],1]),x]))
              
        ################################################w###################################
        #                                  END OF YOUR CODE                               #
        ###################################################################################

        # Save the cost J in every iteration    
        J = cost(x, y, theta)
        
        if (iter+1)%100 is 0:
            print('cost at %d iterations : %f' %(iter+1, J))
        
    return theta

# initialize values
theta = np.zeros(2)
num_iters = 2000
alpha = 0.0001

# compute initial cost
init_J = cost(data_x, data_y, theta)
print('initial cost : %f' %init_J)

# excute gradient descent
theta = gradient_descent(data_x, data_y, theta, alpha, num_iters)

# compute new cost
new_J = cost(data_x, data_y, theta)
print('updated theta : ', theta)
print('updated cost : %f' %new_J)

