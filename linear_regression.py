import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
X - Acidity of Milk
Y - Density of Milk
"""


def readData(filename):
    df = pd.read_csv(filename)
    return df.values


# This function will return array of values
# in the file.

def hypothesis(theta, x):
    return theta[0] + theta[1] * x


def gradient(Y, X, theta):
    grad = np.array([0.0,0.0])
    m = X.shape[0]
    for i in range(m):
        grad[0] += -1*(Y[i] - hypothesis(theta,X[i]))
        grad[1] += -1*(Y[i] - hypothesis(theta,X[i]))*X[i]

    return grad


def gradientDescent(X,Y,learning_rate,maxItr):
    theta = np.array([0.0,0.0])
    e = []
    for i in range(maxItr):
        grad = gradient(Y,X,theta)
        current_error = error(X,Y,theta)
        print(current_error)
        theta[0] = theta[0] - learning_rate*grad[0]
        theta[1] = theta[1] - learning_rate*grad[1]
        e.append(current_error)

    return theta,e


def error(X, Y, theta):
    total_error = 0
    m = X.shape[0]  # That is 99

    for i in range(m):
        total_error += (Y[i] - hypothesis(theta, X[i])) ** 2

    return 0.5 * total_error


x = readData('linearX.csv')
y = readData('linearY.csv')
x.reshape(99, )
y.reshape(99, )
# Normalization : only done at X not Y
x = x - x.mean() / (x.std())

X = x
Y = y

theta,e = gradientDescent(X,Y,learning_rate=0.001,maxItr=560)
print(theta)
print(e)

plt.scatter(X,Y)
plt.plot(X,hypothesis(theta,x),color = 'r')
plt.show()
plt.plot(e,color = 'y')
plt.show()