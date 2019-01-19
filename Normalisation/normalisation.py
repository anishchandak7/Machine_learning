import pandas as pd
import matplotlib.pyplot as plt

data_X = pd.read_csv('linearX.csv')
data_Y = pd.read_csv('linearY.csv')
X = data_X.values
Y = data_Y.values
plt.scatter(X,Y,label='Original Data',color='blue')
#Formula Used
X2 = (X - X.mean())/X.std()
plt.scatter(X2,Y,label='Normalised Data',color='green')
plt.legend()
plt.show()
