import matplotlib.pyplot as plt
import numpy as np

mean1 = [2, 4]
mean2 = [-1, -3]
cov_1 = np.array([[1, 0.8],
                  [0.8, 1]])
cov_2 = np.array([[1, -0.8],
                  [-0.8, 1]])

X1 = np.random.multivariate_normal(mean1, cov_1, 300)
X2 = np.random.multivariate_normal(mean2, cov_2, 400)
plt.scatter(X1[:, 0], X1[:, 1], color="green")
plt.scatter(X2[:, 0], X2[:, 1], color="red")
plt.show()
# See Figure 1
y = np.zeros(700)
y[:300] = 1
X = np.vstack((X1, X2))
plt.scatter(X[:, 0], X[:, 1])
plt.show()
# See figure 2
for i in range(700):
    if y[i] == 1:
        plt.scatter(X[i, 0], X[i, 1], color="red")
    else:
        plt.scatter(X[i, 0], X[i, 1], color="yellow")

# plt.scatter(X[:,0],X[:,1],c=y) THIS LINE OF CODE IS THE REPLACEMENT CODE FOR THE ABOVE FOR - LOOP.
plt.show()
# See Figure 3
