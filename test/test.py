import numpy as np
import pandas as pd


data = pd.read_csv("test.csv", delimiter=" ", header=None, names=['A', 'B', 'C', 'D'])
# print(data.head())
print(data.shape)
x = np.array(data)
print(x)
# dis woooooorks
# x = x.reshape(8, 4, 1)
x = np.reshape(x, (8, 4, 1))
x = np.reshape(x, (8, 4))
print(x)