from __future__ import print_function, division
import torch
import torch.nn as nn                   # All neural network models, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim             # For all optimization algoritms, SGD, Adam, etc.
from torch.optim import lr_scheduler    # To change (update) the learning rate.
import torch.nn.functional as F         # All functions that don't have any parameters.
import numpy as np
import torchvision
from torchvision import datasets        # Has standard datasets that we can import in a nice way.
from torchvision import models
from torchvision import transforms      # Transormations we can perform on our datasets.
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
import matplotlib
from datetime import datetime as dt
import matplotlib.gridspec as gridspec
from pandas import DataFrame
from numpy import vstack
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.optim import Adam
from torch.nn import BCELoss
# torch.set_grad_enabled(True)
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from matplotlib import cm
import seaborn as sns
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error, r2_score

# Read files
def read_csv(directory, file_csv):
    file = pd.read_csv('def_code/datasets/'+directory+'/'+file_csv, encoding='latin1')
    return file

# Medium_office
medium_office_2_100 = read_csv(directory='medium_office', file_csv='Medium_office_2_100.csv')
medium_office_2_100_random_60_perc = read_csv(directory='medium_office', file_csv='Medium_office_2_100_random_60_perc.csv')
medium_office_2_dataset_validation = read_csv(directory='medium_office', file_csv='Medium_office_2_dataset_validation.csv')
medium_office_2_random_2 = read_csv(directory='medium_office', file_csv='Medium_office_2_random_2.csv')

# Small_office
small_office_100 = read_csv(directory='small_office', file_csv='Small_office_100.csv')
small_office_100_random_potenza_60_perc = read_csv(directory='Small_office', file_csv='Small_office_100_random_potenza_60_perc.csv')
small_office_105 = read_csv(directory='small_office', file_csv='Small_office_105.csv')
small_office_random = read_csv(directory='small_office', file_csv='Small_office_random.csv')
"""
# Restaurant
restaurant_100 = read_csv(directory='restaurant', file_csv='Restaurant_100.csv')
restaurant_100_potenza_random_60_percento = read_csv(directory='restaurant', file_csv='Restaurant_100_potenza_random_60_percento.csv')
restaurant_dataset_validation = read_csv(directory='restaurant', file_csv='Restaurant_dataset_validation.csv')
restaurant_random = read_csv(directory='restaurant', file_csv='Restaurant_random.csv')

# Retail
retail_100 = read_csv(directory='retail', file_csv='Retail_100.csv')
retail_100_potenza_random_60_percento = read_csv(directory='retail', file_csv='Retail_100_potenza_random_60_percento.csv')
retail_105 = read_csv(directory='retail', file_csv='Retail_105.csv')
retail_random = read_csv(directory='retail', file_csv='Retail_random.csv')
"""

# Chaining of the datasets
columns = ['Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)', 'Environment:Site Direct Solar Radiation Rate per Area [W/m2](Hourly)',
'Environment:Site Day Type Index [](Hourly)', 'Total Cooling Rate [W]', 'Mean air Temperature [°C]']

def concat_datasets(list, columns):
    name = pd.DataFrame()
    for x in list:
        name = name.append(x[columns], ignore_index=True)
    return name

medium_office = pd.DataFrame()
small_office = pd.DataFrame()
restaurant = pd.DataFrame()
retail = pd.DataFrame()

medium_office = concat_datasets(list=[medium_office_2_100, medium_office_2_100_random_60_perc, medium_office_2_dataset_validation, medium_office_2_random_2], columns=columns) # name=medium_office
small_office = concat_datasets(list=[small_office_100, small_office_100_random_potenza_60_perc, small_office_105, small_office_random], columns=columns)#  name=small_office
# restaurant = concat_datasets(list=[restaurant_100, restaurant_100_potenza_random_60_percento, restaurant_dataset_validation, restaurant_random], columns=columns, name=restaurant)
# retail = concat_datasets(list=[retail_100, retail_100_potenza_random_60_percento, retail_105, retail_random], columns=columns, name=restaurant)

maxT = medium_office['Mean air Temperature [°C]'].max()
minT = medium_office['Mean air Temperature [°C]'].min()

def normalization(df):
    df = (df - df.min()) / (df.max() - df.min())
    # df = (df[col_names] - df[col_names].min()) / (df[col_names].max() - df[col_names].min())
    return df

medium_office = normalization(medium_office)
small_office = normalization(small_office)

"""
def visualization(dataset, column):
    len_f = 0
    for i in range(12):
        len_i = len_f * 976
        len_f = 976 * (i+1)
        fig, axs = plt.subplots(3, 4, figsize=(40,20))
        if i==0:
            axs[i].plot(dataset[column][0:976])
        else:
            axs[i].plot(dataset[column][len_i:len_f])
    plt.show()
"""

"""
def visualization(dataset, column, dataset_name):
    fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(3, 4, figsize=(40, 20))
    ax1.plot(dataset[column][0:976])
    ax2.plot(dataset[column][976:1952])
    ax3.plot(dataset[column][1952:2928])
    ax4.plot(dataset[column][2928:3904])
    ax5.plot(dataset[column][3904:4880])
    ax6.plot(dataset[column][4880:5856])
    ax7.plot(dataset[column][5856:6832])
    ax8.plot(dataset[column][6832:7808])
    ax9.plot(dataset[column][7808:8784])
    ax10.plot(dataset[column][8784:9760])
    ax11.plot(dataset[column][9760:10736])
    ax12.plot(dataset[column][10736:11712])
    fig.suptitle(dataset_name+': '+column, size=50)
    plt.show()
    if column == 'Mean air Temperature [°C]':
        fig.savefig('def_code/immagini/mean_air_temperature/{}_{}.png'.format(dataset_name, column))
    if column == 'Total Cooling Rate [W]':
        fig.savefig('def_code/immagini/total_cooling_rate/{}_{}.png'.format(dataset_name, column))

# Plot mean air temperature
visualization(medium_office, 'Mean air Temperature [°C]', dataset_name='Medium office')
visualization(small_office, 'Mean air Temperature [°C]', dataset_name='Small office')
#visualization(restaurant, 'Mean air Temperature [°C]', dataset_name='Restaurant')
#visualization(retail, 'Mean air Temperature [°C]', dataset_name='Retail')

# Plot the total cooling rate
visualization(medium_office, 'Total Cooling Rate [W]', dataset_name='Medium office')
visualization(small_office, 'Total Cooling Rate [W]', dataset_name='Small office')
#visualization(restaurant, 'Total Cooling Rate [W]', dataset_name='Restaurant')
#visualization(retail, 'Total Cooling Rate [W]', dataset_name='Retail')
"""
#______________________________________Datasets_preprocessing___________________________________________________________
# shifting_period = 1
period = 1
l_train = int(0.8 * len(medium_office))
l_train_m = int(0.8 * l_train)# training length

def create_data(df, col_name):
    train_mx = pd.DataFrame(df[:l_train_m])
    val_mx = pd.DataFrame(df[l_train_m:l_train])
    test_mx = pd.DataFrame(df[l_train:])
    train_mx['out'] = train_mx[col_name]
    val_mx['out'] = val_mx[col_name]
    test_mx['out'] = test_mx[col_name]
    train_mx[col_name] = train_mx[col_name].shift(periods=period) # shifting train_x
    val_mx[col_name] = val_mx[col_name].shift(periods=period)
    test_mx[col_name] = test_mx[col_name].shift(periods=period)
    train_mx = train_mx.iloc[period:] # delete the Nan
    val_mx = val_mx.iloc[period:]
    test_mx = test_mx.iloc[period:]
    train_mx = train_mx.reset_index(drop=True) # reset the index of the rows
    val_mx = val_mx.reset_index(drop=True)
    test_mx = test_mx.reset_index(drop=True)
    return train_mx, val_mx, test_mx

train_m, val_m, test_m = create_data(df=medium_office, col_name='Mean air Temperature [°C]')
train_s, val_s, test_s = create_data(df=small_office, col_name='Mean air Temperature [°C]')
train_m, val_m, test_m = train_m.to_numpy(), val_m.to_numpy(), test_m.to_numpy()
train_s, val_s, test_s = train_s.to_numpy(), val_s.to_numpy(), test_s.to_numpy()

#_____________________________________Split_the_x_and_y_datasets________________________
n_steps = 48
# train_m = train_m[:7488]

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        # seq_y = sequences[end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Split medium office
train_mX, train_mY = split_sequences(sequences=train_m, n_steps=n_steps)
val_mX, val_mY = split_sequences(sequences=val_m, n_steps=n_steps)
test_mX, test_mY = split_sequences(sequences=test_m, n_steps=n_steps)

# Split small office
train_sX, train_sY = split_sequences(sequences=train_s, n_steps=n_steps)
val_sX, val_sY = split_sequences(sequences=val_s, n_steps=n_steps)
test_sX, test_sY = split_sequences(sequences=test_s, n_steps=n_steps)

# Convert medium office to tensors
train_mX = torch.from_numpy(train_mX)
train_mY = torch.from_numpy(train_mY)
val_mX = torch.from_numpy(val_mX)
val_mY = torch.from_numpy(val_mY)
test_mX = torch.from_numpy(test_mX)
test_mY = torch.from_numpy(test_mY)

# Convert small office to tensors
train_sX = torch.from_numpy(train_sX)
train_sY = torch.from_numpy(train_sY)
val_sX = torch.from_numpy(val_sX)
val_sY = torch.from_numpy(val_sY)
test_sX = torch.from_numpy(test_sX)
test_sY = torch.from_numpy(test_sY)

print(type(train_mX), train_mX.shape)
print(type(train_mY), train_mY.shape)
print(type(val_mX), val_mX.shape)
print(type(val_mY), val_mY.shape)
print(type(test_mX), test_mX.shape)
print(type(test_mY), test_mY.shape)

print(type(train_sX), train_sX.shape)
print(type(train_sY), train_sY.shape)
print(type(val_sX), val_sX.shape)
print(type(val_sY), val_sY.shape)
print(type(test_sX), test_sX.shape)
print(type(test_sY), test_sY.shape)
