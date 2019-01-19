import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('mnist_train.csv')
#print(data.head(n=5))
data = df.values
X = data[:,1:]
Y = data[:,0]
#print(X.shape)
#print(Y.shape)

#plt.imshow(X,cmap="gray")

img = X[4].reshape((28,28))
print(Y[4])
plt.imshow(img,cmap="gray")
plt.show()

split = int(0.8*X.shape[0])
print(split)

X_train,Y_train = X[:split,:],Y[:split]
X_test,Y_test = X[split:,:],Y[split:]
print(X_train,Y_train)
print(X_test,Y_test)