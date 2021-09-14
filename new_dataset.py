import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime as dt
import os
import numpy as np
from numpy.random import randn
import matplotlib.gridspec as gridspec
from pandas import DataFrame
from pandas import concat
import torch
import torch.nn as nn
from torch import nn
from numpy import vstack
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import lr_scheduler    # To change (update) the learning rate.
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
import torch.nn.functional as F
import torch.optim as optim
import torchvision
torch.set_grad_enabled(True)
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from matplotlib import cm
import seaborn as sns
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('ASHRAE90.1_OfficeSmall_STD2016_NewYork.csv')
df = df.iloc[288:,:]
Date = df['Date/Time'].str.split(' ', expand=True)
Date.rename(columns={0:'nullo',1:'date',2:'null', 3:'time'},inplace=True)
Date['time'] = Date['time'].replace(to_replace='24:00:00', value= '0:00:00')
data = Date['date']+' '+Date['time']
data = pd.to_datetime(data, format='%m/%d %H:%M:%S')

df['day'] = data.apply(lambda x: x.day)
df['month'] = data.apply(lambda x: x.month)
df['hour'] = data.apply(lambda x: x.hour)
df['dn'] = data.apply(lambda x: x.weekday())
df['data'] = Date.date

df = df.reset_index(drop=True)

# create the list of input columns
col_names = ['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)',
             'Environment:Site Diffuse Solar Radiation Rate per Area [W/m2](TimeStep)',
             'CORE_ZN:Zone People Occupant Count [](TimeStep)',
             'PSZ-AC:1 HEAT PUMP DX COOLING COIL:Cooling Coil Total Cooling Rate [W](TimeStep)',
             'PSZ-AC:1 HEAT PUMP DX HEATING COIL:Heating Coil Heating Rate [W](TimeStep)',
             'CORE_ZN:Zone Mean Air Temperature [C](TimeStep)']

multi_norm = (df[col_names]-df[col_names].min())/(df[col_names].max()-df[col_names].min())

# df['random_temp'] = np.sort(np.random(0.5, -0.5, len(df)))
random_num = []
for x in range(len(multi_norm)):
    random_num.append(np.random.uniform(-0.5, 0.5))

random_num = pd.Series(random_num)
multi_norm['random_num'] = random_num
multi_norm['random_temp'] = multi_norm['CORE_ZN:Zone Mean Air Temperature [C](TimeStep)'] + random_num

#_______DELETE_USELESS_COLUMNS________________
multi_norm.drop('random_num', inplace=True, axis=1)
multi_norm.drop('CORE_ZN:Zone Mean Air Temperature [C](TimeStep)', inplace=True, axis=1)


shifting_period = 1
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
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)



l_train = int(0.8 * len(multi_norm))
l_train_m = int(0.8 * l_train)# training length
# l_val_m = int(0.2*l_train)# validation length
maxT = df['CORE_ZN:Zone Mean Air Temperature [C](TimeStep)'].max()  # max value
minT = df['CORE_ZN:Zone Mean Air Temperature [C](TimeStep)'].min()  # max value


def multi_shift(df, col_name):
    train_mx = pd.DataFrame(df[:l_train_m]) # creating train_x dataset
    val_mx = pd.DataFrame(df[l_train_m:l_train]) # creating val_x dataset
    test_mx = pd.DataFrame(df[l_train:]) # creating test_x dataset
    train_mx[col_name] = train_mx[col_name].shift(periods=period) # shifting train_x
    val_mx[col_name] = val_mx[col_name].shift(periods=period) # shifting val_x
    test_mx[col_name] = test_mx[col_name].shift(periods=period) # shifting test_x
    train_mx = train_mx.reset_index(drop=True) # reset the index of the rows
    val_mx = val_mx.reset_index(drop=True) # reset the index of the rows
    test_mx = test_mx.reset_index(drop=True) # reset the index of the rows
    train_mx = train_mx.iloc[period:] # delete the Nan
    val_mx = val_mx.iloc[period:]
    test_mx = test_mx.iloc[period:] # delete the Nan
    return train_mx, test_mx, val_mx

period = shifting_period
train_mx, test_mx, val_mx = multi_shift(multi_norm, col_name='random_temp')
train_my = (df['CORE_ZN:Zone Mean Air Temperature [C](TimeStep)'][1:l_train_m]-minT)/(maxT-minT)
val_my = (df['CORE_ZN:Zone Mean Air Temperature [C](TimeStep)'][l_train_m+1:l_train]-minT)/(maxT-minT)
test_my = (df['CORE_ZN:Zone Mean Air Temperature [C](TimeStep)'][l_train+1:]-minT)/(maxT-minT)
test_my = test_my.reset_index(drop=True)
test_mx = test_mx.reset_index(drop=True)
val_my = val_my.reset_index(drop=True)
val_mx = val_mx.reset_index(drop=True)
train_m = train_mx
val_m = val_mx
test_m = test_mx
train_m['out'] = train_my
val_m['out'] = val_my
test_m['out'] = test_my
train_m = train_m.to_numpy()
val_m = val_m.to_numpy()
test_m = test_m.to_numpy()


# Split the x and y datasets
#======================================================
n_steps = 48
train_mX, train_mY = split_sequences(train_m, n_steps=n_steps)
val_mX, val_mY = split_sequences(val_m, n_steps=n_steps)
test_mX, test_mY = split_sequences(test_m, n_steps=n_steps)

# Convert to tensors
train_mX_new = torch.from_numpy(train_mX)
train_mY_new = torch.from_numpy(train_mY)
val_mX_new = torch.from_numpy(val_mX)
val_mY_new = torch.from_numpy(val_mY)
test_mX_new = torch.from_numpy(test_mX)
test_mY_new = torch.from_numpy(test_mY)

print(type(train_mX_new), train_mX_new.shape)
print(type(train_mY_new), train_mY_new.shape)
print(type(val_mX_new), val_mX_new.shape)
print(type(val_mY_new), val_mY_new.shape)
print(type(test_mX_new), test_mX_new.shape)
print(type(test_mY_new), test_mY_new.shape)

