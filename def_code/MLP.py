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
medium_office_2_dataset_validation = read_csv(directory='medium_office', file_csv='Medium_office_2_dataset_validation.csv')
medium_office_2_random_2 = read_csv(directory='medium_office', file_csv='Medium_office_2_random_2.csv')
medium_office_2_100_random_60_perc = read_csv(directory='medium_office', file_csv='Medium_office_2_100_random_60_perc.csv')

# Chaining of the datasets
columns = ['Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)', 'Environment:Site Direct Solar Radiation Rate per Area [W/m2](Hourly)',
'Environment:Site Day Type Index [](Hourly)', 'Total Cooling Rate [W]', 'Total People', 'Mean air Temperature [°C]']

def concat_datasets(list, columns):
    name = pd.DataFrame()
    for x in list:
        name = name.append(x[columns], ignore_index=True)
    for i in range(0, len(name)):
        if name['Total People'][i] != 0.00000:
            name['Total People'][i] = 1.0
    return name

medium_office = pd.DataFrame()

medium_office = concat_datasets(list=[medium_office_2_100, medium_office_2_dataset_validation, medium_office_2_random_2,medium_office_2_100_random_60_perc], columns=columns) # name=medium_office

# ___________________________________________________Normalization______________________________________________________

maxT_m = medium_office['Mean air Temperature [°C]'].max()
minT_m = medium_office['Mean air Temperature [°C]'].min()
# maxT_s = small_office['Mean air Temperature [°C]'].max()
# minT_s = small_office['Mean air Temperature [°C]'].min()

def normalization(df, mode):
    if mode == 'norm':
        x = (df - df.min()) / (df.max() - df.min())
    if mode == 'T_denorm':
        x = minT_m + df*(maxT_m-minT_m)
    return x

medium_office = normalization(medium_office, mode='norm')

medium_office['Environment:Site Day Type Index [](Hourly)'] = round(medium_office['Environment:Site Day Type Index [](Hourly)'], 2)


# ______________________________________Datasets_preprocessing__________________________________________________________
period = 1
l_train = int(0.5 * len(medium_office))
l_val = int(l_train+2928)
# l_test = int(len(medium_office)-l_val)
# l_train_m = int(0.8 * l_train)# training length

def create_data(df, col_name):
    train_m = pd.DataFrame(df[:l_train])
    val_m = pd.DataFrame(df[l_train:l_val])
    test_m = pd.DataFrame(df[l_val:])
    train_m[col_name] = train_m[col_name].shift(periods=period) # shifting train_x
    val_m[col_name] = val_m[col_name].shift(periods=period)
    test_m[col_name] = test_m[col_name].shift(periods=period)
    train_m = train_m.iloc[period:] # delete the Nan
    val_m = val_m.iloc[period:]
    test_m = test_m.iloc[period:]
    train_m = train_m.reset_index(drop=True) # reset the index of the rows
    val_m = val_m.reset_index(drop=True)
    test_m = test_m.reset_index(drop=True)
    return train_m, val_m, test_m

train_m, val_m, test_m = create_data(df=medium_office, col_name='Mean air Temperature [°C]')
train_m, val_m, test_m = train_m.to_numpy(), val_m.to_numpy(), test_m.to_numpy()


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


n_steps = 4
train_mX, train_mY = split_sequences(train_m, n_steps=n_steps)
test_mX, test_mY = split_sequences(test_m, n_steps=n_steps)


# Convert to tensors
train_mX = torch.from_numpy(train_mX)
train_mY = torch.from_numpy(train_mY)
val_mX = torch.from_numpy(test_mX)
val_mY = torch.from_numpy(test_mY)
test_mX = torch.from_numpy(test_mX)
test_mY = torch.from_numpy(test_mY)

print(type(train_mX), train_mX.shape)
print(type(train_mY), train_mY.shape)
print(type(val_mX), val_mX.shape)
print(type(val_mY), val_mY.shape)
print(type(test_mX), test_mX.shape)
print(type(test_mY), test_mY.shape)

# ________________________________________________MLP NETWORK ___________________________________________________________
n_features_m = 20
# Multivariate model definition
class MLP_m(nn.Module):
    # define model elements
    def __init__(self, n_features_m):
        super(MLP_m, self).__init__()
        self.hidden1 = Linear(n_features_m, 100) # input to first hidden layer
        self.act1 = ReLU()
        self.hidden2 = Linear(100, 100)
        self.act2 = ReLU()
        self.hidden3 = Linear(100, 100)
        self.act3 = ReLU()
        self.hidden4 = Linear(100, 100)
        self.act4 = ReLU()
        self.hidden5 = Linear(100, 100)
        self.act5 = ReLU()
        self.hidden6 = Linear(100, 1)

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        X = self.act3(X)
        # fourth hidden layer and output
        X = self.hidden4(X)
        X = self.act4(X)
        # fifth hidden layer and output
        X = self.hidden5(X)
        X = self.act5(X)
        # sexth hidden layer and output
        X = self.hidden6(X)

        return X


train_batch_size_m = 400
train_data_m = TensorDataset(train_mX, train_mY)
train_dl_m = DataLoader(train_data_m, batch_size=train_batch_size_m, shuffle=True)

val_batch_size_m = 200
val_data_m = TensorDataset(val_mX, val_mY)
val_dl_m = DataLoader(val_data_m, batch_size=val_batch_size_m, shuffle=True)

test_data_m = TensorDataset(test_mX, test_mY)
test_dl_m = DataLoader(test_data_m) # batch_size=400

mlp_m = MLP_m(n_features_m)
optimizer = optim.Adam(mlp_m.parameters(), lr=0.008)


train_loss_m = []
val_loss_m = []
epochs = 100

# Training with multiple epochs
for epoch in range(epochs):
    # ________________TRAINING_______________________________
    total_loss = 0
    for x, y in train_dl_m: # get batch
        input = x.reshape(-1, n_features_m)
        output = mlp_m(input.float())
        loss = F.mse_loss(output.view(-1), y.float()) # calculate the loss

        loss.backward() # calculate the gradient
        optimizer.step() # update weight
        optimizer.zero_grad()

        total_loss += loss.item()
    train_loss_m.append(total_loss)
    # total_correct += get_num_correct(preds, labels)
    print("epoch: ", epoch, "loss: ", total_loss/train_batch_size_m)

    # ________________VALIDATION__________________________
    valid_total_loss_m = 0
    for n, m in val_dl_m:
        input = n.reshape(-1, n_features_m)
        output = mlp_m(input.float())
        v_loss_m = F.mse_loss(output.view(-1), m.float())  # calculate the loss

        valid_total_loss_m += v_loss_m.item()
    val_loss_m.append(valid_total_loss_m)
    print("epoch: ", epoch, "validation loss: ", valid_total_loss_m/val_batch_size_m)

"""
torch.save(mlp_m.state_dict(), "def_code/MLP_20_100x5.pth")
loadedmodel_m = MLP_m(n_features_m)
loadedmodel_m.load_state_dict(torch.load("MLP_20_100x5.pth"))
loadedmodel_m.eval()
"""

# plot in log
plt.plot(train_loss_m, c='b', label='Train loss')
plt.plot(val_loss_m, c='r', label='Validation loss')
plt.grid()
#plt.yscale("log")
plt.title('Loss value trend', size=15)
plt.xlabel('Epochs')
plt.legend()
#plt.savefig('def_code/immagini/MLP/20_100x5/Loss_value_trend({}_epochs).png'.format(epochs))
plt.show()


# Evaluate the testing set
def evaluate_model(test_dl, model, n_features):
    predictions, actuals = list(), list()
    # test_loss = []
    for x, targets in test_dl:
        total_loss = 0
        # evaluate the model on the test set
        inputs = x.reshape(-1, n_features)
        yhat = model(inputs.float())
        loss = F.mse_loss(yhat.view(-1), targets.float())
        # retrieve numpy array
        # yhat = yhat.detach().numpy()
        # actual = targets.numpy()
        # actual = targets.reshape((len(targets), -1))
        # round to class values
        # yhat = yhat.round()
        # store
        total_loss += loss.item()
        #test_loss.append(total_loss)
        predictions.append(yhat.item())
        actuals.append(targets.item())

    return predictions, actuals


y_pred_m, y_lab_m = evaluate_model(test_dl_m, mlp_m, n_features=n_features_m)
y_pred_m = np.array(y_pred_m)
y_lab_m = np.array(y_lab_m)

y_pred_m = normalization(y_pred_m, mode='T_denorm')
y_lab_m = normalization(y_lab_m, mode='T_denorm')

"""
flatten = lambda l: [item for sublist in l for item in sublist]
y_pred_m = flatten(y_pred_m)
y_lab_m = flatten(y_lab_m)
"""

y_pred_m = np.array(y_pred_m, dtype=float)
y_lab_m = np.array(y_lab_m, dtype=float)


error_m = []
error_m = y_pred_m - y_lab_m


plt.plot(y_lab_m, c='r', label='Actual')
plt.plot(y_pred_m, c='b', label='Predicted')
plt.xlim(0, 600)
plt.title('Predictions vs Real trend (Multivariate case)', size=15)
plt.legend()
#plt.savefig('def_code/immagini/MLP/20_100x5/Predictions_vs_Real_values({}_epochs).png'.format(epochs))
plt.show()


plt.hist(error_m, 150, linewidth=1.5, edgecolor='black', color='orange')
plt.xticks(np.arange(-0.4, 0.4, 0.1))
plt.xlim(-0.4, 0.4)
plt.title('LSTM model prediction error')
# plt.xlabel('Error')
plt.grid(True)
#plt.savefig('def_code/immagini/MLP/20_100x5/LSTM_model_error({}_epochs).png'.format(epochs))
plt.show()



# METRICS
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


MAPE = mean_absolute_percentage_error(y_lab_m, y_pred_m)
MSE = mean_squared_error(y_lab_m, y_pred_m)
R2 = r2_score(y_lab_m, y_pred_m)

print('MAPE:%0.5f%%'%MAPE)
print('MSE:', MSE.item())
print('R2:', R2.item())

plt.scatter(y_lab_m, y_pred_m,  color='k', edgecolor= 'white', linewidth=1) # ,alpha=0.1
plt.text(23.2, 28.2, 'MAPE: {:.3f}'.format(MAPE), fontsize=15, bbox=dict(facecolor='red', alpha=0.5))
plt.text(23.2, 29.2, 'MSE: {:.3f}'.format(MSE), fontsize=15, bbox=dict(facecolor='green', alpha=0.5))
plt.plot([23, 27, 30], [23, 27, 30], color='red')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlabel('Real Temperature [°C]')
plt.ylabel('Predicted Temperature [°C]')
plt.title("Tuning prediction distribution", size=15)
#plt.savefig('def_code/immagini/MLP/20_100x5/LSTM_tuning_prediction_distribution({}_epochs).png'.format(epochs))
plt.show()






