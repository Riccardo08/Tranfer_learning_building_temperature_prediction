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

def normalization(df):
    df = (df - df.min()) / (df.max() - df.min())
    return df

medium_office = normalization(medium_office)

medium_office['Environment:Site Day Type Index [](Hourly)'] = round(medium_office['Environment:Site Day Type Index [](Hourly)'], 2)


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


n_steps=48
train_mX, train_mY = split_sequences(train_m, n_steps=n_steps)
test_mX, test_mY = split_sequences(test_m, n_steps=n_steps)









n_features_m = 288
# Multivariate model definition
class MLP_m(nn.Module):
    # define model elements
    def __init__(self, n_features_m):
        super(MLP_m, self).__init__()
        self.hidden1 = Linear(n_features_m, 150) # input to first hidden layer
        self.act1 = ReLU()
        self.hidden2 = Linear(150, 70) # second hidden layer
        self.act2 = ReLU()
        self.hidden3 = Linear(70, 1) # third hidden layer and output

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
        return X

train_batch_size_m = 700
train_data_m = TensorDataset(train_mX, train_mY)
train_dl_m = DataLoader(train_data_m, batch_size=train_batch_size_m, shuffle=True)

val_batch_size_m = 500
val_data_m = TensorDataset(val_mX, val_mY)
val_dl_m = DataLoader(val_data_m, batch_size=val_batch_size_m, shuffle=True)

test_data_m = TensorDataset(test_mX, test_mY)
test_dl_m = DataLoader(test_data_m) # batch_size=400

mlp_m = MLP_m(n_features_m)
optimizer = optim.Adam(mlp_m.parameters(), lr=0.001) # lr "learning rate"


train_loss_m = []
val_loss_m = []
# Training with multiple epochs
for epoch in range(35):
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

    valid_total_loss_m = 0
    for n, m in val_dl_m:
        input = n.reshape(-1, n_features_m)
        output = mlp_m(input.float())
        v_loss_m = F.mse_loss(output.view(-1), m.float())  # calculate the loss

        valid_total_loss_m += v_loss_m.item()
    val_loss_m.append(valid_total_loss_m)
    print("epoch: ", epoch, "validation loss: ", valid_total_loss_m/val_batch_size_m)


# total_correct/len(train_set)
torch.save(mlp_m.state_dict(), "rete_neurale_m.pth")
loadedmodel_m = MLP_m(n_features_m)
loadedmodel_m.load_state_dict(torch.load("rete_neurale_m.pth"))
loadedmodel_m.eval()


# Evaluate the testing set
def evaluate_model(test_dl, model, n_features):
    predictions, actuals= list(), list()
    #test_loss = []
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
        predictions.append(yhat)
        actuals.append(targets.item())
        # gap.append(targets-yhat)
    return predictions, actuals


predictions_m, actuals_m = evaluate_model(test_dl_m, loadedmodel_m, n_features=n_features_m)
predictions_m = np.array(predictions_m)
actuals_m = np.array(actuals_m)

# Normalization
def denormalization(x, max_value):
    x = x * max_value
    return x

predictions_m = denormalization(predictions_m, max_value=max)
actuals_m = denormalization(actuals_m, max_value=max)

# PLOTTING
sns.set_style("darkgrid")
plt.scatter(actuals_m, predictions_m, color='gold', alpha=0.3, edgecolors='k')
plt.plot([0.60, 0.75, 1], [0.60, 0.75, 1])
plt.xlim(np.min(actuals_m), np.max(actuals_m))
plt.xlabel('True Values ')
plt.ylabel('Predicted values ')
plt.title("Evaluation of testing dataset (Multivariate case)", size =18)
plt.savefig('Evaluation_of_testing_dataset_(Multivariate_case).png')
plt.show()

plt.plot(actuals_m, c='r', label='Actual')
plt.plot(predictions_m, c='b', label='Predicted')
plt.xlim(0, 600)
plt.title('Predictions vs Real trend (Multivariate case)', size=15)
plt.legend()
plt.savefig('Predictions_vs_Real_trend_(Multivariate_case).png')
plt.show()


# plot in log
plt.plot(train_loss_m, c='b', label='Train loss')
plt.plot(val_loss_m, c='r', label='Validation loss')
plt.yscale("log")
plt.title('Loss value trend (Multivariate case)', size=15)
plt.xlabel('Epochs')
plt.legend()
plt.savefig('Loss_value_trend_(Multivariate_case).png')
plt.show()

























