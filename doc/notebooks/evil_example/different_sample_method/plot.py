import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


# plot max error
path = './sample_with_1000_points'
files = os.listdir(path)
for file in files:
    if 'k=12.5' in file:
        plt.title('different sample method when k=12.5')
        plt.ylabel('maxmin error')
        plt.xlabel('selected number points')
        data = pd.read_csv('./sample_with_1000_points/{}'.format(file),index_col=0)
        plt.plot(np.arange(1,1000,5),data['max_error'][np.arange(1,1000,5)],label='{}'.format(file))

plt.legend()


# plot mean absolute error
path = './sample_with_1000_points'
files = os.listdir(path)
for file in files:
    if 'k=12.5' in file:
        plt.title('different sample method when k=12.5')
        plt.ylabel('mean absolute error')
        plt.xlabel('selected number points')
        data = pd.read_csv('./sample_with_1000_points/{}'.format(file),index_col=0)
        plt.plot(np.arange(1,1000,2),data['mean_absolute_error'][np.arange(1,1000,2)],label='{}'.format(file))

plt.legend()


# plot mean squared error
path = './sample_with_1000_points'
files = os.listdir(path)
for file in files:
    if 'k=0.0' in file:
        plt.title('different sample method when k=0.0')
        plt.ylabel('mean squared error')
        plt.xlabel('selected number points')
        data = pd.read_csv('./sample_with_1000_points/{}'.format(file),index_col=0)
        plt.plot(np.arange(1,1000,5),data['mean_squared_error'][np.arange(1,1000,5)],label='{}'.format(file))

plt.legend()
