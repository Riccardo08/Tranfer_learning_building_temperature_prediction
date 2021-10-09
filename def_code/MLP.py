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
#l_train = int(0.5 * len(medium_office))
#l_val = int(l_train+2928)
# l_test = int(len(medium_office)-l_val)
# l_train_m = int(0.8 * l_train)# training length

def create_data(df, col_name):
    l_train = int(0.5 * len(df))
    l_val = int(l_train + int(len(df)/4))
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


def train_model(model, epochs, train_dl, val_dl, optimizer, train_batch_size, val_batch_size, mode=''):
    model.train()
    train_loss = []
    val_loss = []
    # Training with multiple epochs
    for epoch in range(epochs):
        # ________________TRAINING_______________________________
        total_loss = 0
        for x, y in train_dl: # get batch
            input = x.reshape(-1, n_features_m)
            output = model(input.float())
            loss = F.mse_loss(output.view(-1), y.float()) # calculate the loss

            loss.backward() # calculate the gradient
            optimizer.step() # update weight
            optimizer.zero_grad()

            total_loss += loss.item()
        train_loss.append(total_loss)
        # total_correct += get_num_correct(preds, labels)
        if mode == 'tuning':
            lr_scheduler.step()
        print("epoch: ", epoch, "loss: ", total_loss/train_batch_size)

        # ________________VALIDATION_____________________________
        valid_total_loss = 0
        for n, m in val_dl:
            input = n.reshape(-1, n_features_m)
            output = model(input.float())
            v_loss_m = F.mse_loss(output.view(-1), m.float())  # calculate the loss
            valid_total_loss += v_loss_m.item()
        val_loss.append(valid_total_loss)
        print("epoch: ", epoch, "validation loss: ", valid_total_loss/val_batch_size)
    return train_loss, val_loss

epochs = 500
train_loss_m, val_loss_m = train_model(mlp_m, epochs, train_dl_m, val_dl_m, optimizer, train_batch_size_m, val_batch_size_m)


#torch.save(mlp_m.state_dict(), "def_code/MLP_20_100x5.pth")
model = MLP_m(n_features_m)
model.load_state_dict(torch.load("def_code/MLP_20_100x5.pth"))
model.eval()


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
def evaluate_model(model, test_dl, n_features, maxT, minT):
    model.eval()
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
        yhat = minT + yhat*(maxT-minT)
        targets = minT + targets*(maxT-minT)
        predictions.append(yhat)
        actuals.append(targets)

    return predictions, actuals


y_pred_m, y_lab_m = evaluate_model(mlp_m, test_dl_m, n_features=n_features_m, maxT=maxT_m, minT=minT_m)
y_pred_m = np.array(y_pred_m)
y_lab_m = np.array(y_lab_m)

# y_pred_m = normalization(y_pred_m, mode='T_denorm')
# y_lab_m = normalization(y_lab_m, mode='T_denorm')

"""
flatten = lambda l: [item for sublist in l for item in sublist]
y_pred_m = flatten(y_pred_m)
y_lab_m = flatten(y_lab_m)
y_pred_m = np.array(y_pred_m, dtype=float)
y_lab_m = np.array(y_lab_m, dtype=float)
"""

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



# _____________________________________________________TUNING_PHASE_____________________________________________________

def freeze_params(model, mode=''):
    if mode == 'tail':
        for param_c in model.hidden6.parameters():
            param_c.requires_grad = False
    if mode == 'head':
        for param_c in model.hidden1.parameters():
            param_c.requires_grad = False
    return model


mlp_head = freeze_params(model, mode='head')
mlp_tail = freeze_params(model, mode='tail')

print(mlp_head)
for i in mlp_head.parameters():
    print(i)
for x in mlp_tail.parameters():
    print(x)

# _______________________________________ADD MODULES____________________________________________________________________
"""

num_out_ftrs = lstm_test.l_linear[0].out_features

# lstm_test.l_linear[1] = lstm_test.l_linear.add_module('1', nn.ReLU())
# lstm_test.l_linear[2] = lstm_test.l_linear.add_module('2', nn.Linear(num_out_ftrs, 1))

# lstm_test.l_linear[1] = nn.ReLU()
lstm_test.l_linear[1] = nn.ReLU()
lstm_test.l_linear[2] = nn.Linear(num_out_ftrs, 5)
lstm_test.l_linear.add_module('3', nn.ReLU())
lstm_test.l_linear.add_module('4', nn.Linear(5, 1))

# lstm_test.l_linear[3] = nn.ReLU()
# lstm_test.l_linear[4] = nn.Linear(5, 1)


lstm_test.l_linear.add_module('5', nn.ReLU())
lstm_test.l_linear.add_module('6', nn.Linear(4, 1))


print(lstm_test)
for i in lstm_test.l_lstm.parameters():
    print(i)
for x in lstm_test.l_linear.parameters():
    print(x)
"""

# ____________________________________________________Small office______________________________________________________

#small_office_100 = read_csv(directory='small_office', file_csv='Small_office_100.csv')
small_office_100_random_potenza_60_perc = read_csv(directory='Small_office', file_csv='Small_office_100_random_potenza_60_perc.csv')
# small_office_105 = read_csv(directory='small_office', file_csv='Small_office_105.csv')
# small_office_random = read_csv(directory='small_office', file_csv='Small_office_random.csv')

one_month_small = small_office_100_random_potenza_60_perc[0:720][columns]

maxT_small_1m = one_month_small['Mean air Temperature [°C]'].max()
minT_small_1m = one_month_small['Mean air Temperature [°C]'].min()


def set_binary(df, column):
    for i in range(0, len(df)):
        if df[column][i] != 0.00000:
            df[column][i] = 1.0
    return df

one_month_small = set_binary(one_month_small, 'Total People')

one_month_small = normalization(one_month_small, 'norm')


train_small_1m, val_small_1m, test_small_1m = create_data(df=one_month_small, col_name='Mean air Temperature [°C]')
train_small_1m, val_small_1m, test_small_1m = train_small_1m.to_numpy(), val_small_1m.to_numpy(), test_small_1m.to_numpy()


# Split small office
train_small_1mX, train_small_1mY = split_sequences(sequences=train_small_1m, n_steps=n_steps)
val_small_1mX, val_small_1mY = split_sequences(sequences=val_small_1m, n_steps=n_steps)
test_small_1mX, test_small_1mY = split_sequences(sequences=test_small_1m, n_steps=n_steps)

# Convert small office to tensors
train_small_1mX = torch.from_numpy(train_small_1mX)
train_small_1mY = torch.from_numpy(train_small_1mY)
val_small_1mX = torch.from_numpy(val_small_1mX)
val_small_1mY = torch.from_numpy(val_small_1mY)
test_small_1mX = torch.from_numpy(test_small_1mX)
test_small_1mY = torch.from_numpy(test_small_1mY)

print(type(train_small_1mX), train_small_1mX.shape)
print(type(train_small_1mY), train_small_1mY.shape)
print(type(val_small_1mX), val_small_1mX.shape)
print(type(val_small_1mY), val_small_1mY.shape)
print(type(test_small_1mX), test_small_1mX.shape)
print(type(test_small_1mY), test_small_1mY.shape)


# __________________________________INCLUDE_NEW_DATASET_________________________________________________________________
# from new_dataset import train_mX_new, train_mY_new, val_mX_new, val_mY_new, test_mX_new, test_mY_new
# from one_month_small import train_small_1mX, train_small_1mY, val_small_1mX, val_small_1mY, test_small_1mX, test_small_1mY, maxT_small_1m, minT_small_1m

train_batch_size = 80
train_data_small_1m = TensorDataset(train_small_1mX, train_small_1mY)
train_dl_small_1m = DataLoader(train_data_small_1m, batch_size=train_batch_size, shuffle=True, drop_last=True)

val_batch_size = 18
val_data_small_1m = TensorDataset(val_small_1mX, val_small_1mY)
val_dl_small_1m = DataLoader(val_data_small_1m, batch_size=val_batch_size, shuffle=True, drop_last=True)


# generalize the number of features and the number of timesteps by linking them to the preprocessing
n_features = train_small_1mX.shape[2]
n_timesteps = n_steps

# initialize the network,criterion and optimizer
criterion_ft = torch.nn.MSELoss()
# optimizer_ft = torch.optim.SGD(mlp_m.parameters(), lr=0.001)
optimizer_ft = optim.Adam(mlp_m.parameters(), lr=0.008)

# optimizer_ft = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
# Decay LR (learning rate) by a factor of 0.1 every 7 epochs
lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

# TRAINING TUNING MODEL
epochs_s = 500
train_loss_st_1m, val_loss_st_1m = train_model(mlp_m, epochs_s, train_dl_small_1m, val_dl_small_1m, optimizer_ft, train_batch_size, val_batch_size, mode='')

# Plot to verify validation and train loss, in order to avoid underfitting and overfitting
plt.plot(train_loss_st_1m, '--', color='r', linewidth=1, label='Train Loss')
plt.plot(val_loss_st_1m, color='b', linewidth=1, label='Validation Loss')
plt.ylabel('Loss (MSE)')
#plt.ylim(0, 0.005)
plt.xlabel('Epoch')
plt.xticks(np.arange(0, int(epochs_s), 30))
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.title("Training VS Validation loss", size=15)
plt.legend()
#plt.savefig('def_code/immagini/MLP/20_100x5/train_on_one_month_small/LSTM_tuning_Train_VS_Val_LOSS({}_epochs).png'.format(epochs_s))
plt.show()

# ______________________________________TESTING______________________________
test_batch_size = 18
test_data_s = TensorDataset(test_small_1mX, test_small_1mY)
test_dl_s = DataLoader(test_data_s, shuffle=False, batch_size=test_batch_size, drop_last=True)
test_losses_s = []
# h = lstm.init_hidden(val_batch_size)

y_pred_st_1m, y_lab_st_1m = evaluate_model(mlp_m, test_dl_s, n_features_m, maxT_small_1m, minT_small_1m)

flatten = lambda l: [item for sublist in l for item in sublist]
y_pred_st_1m = flatten(y_pred_st_1m)
y_lab_st_1m = flatten(y_lab_st_1m)
y_pred_st_1m = np.array(y_pred_st_1m, dtype=float)
y_lab_st_1m = np.array(y_lab_st_1m, dtype = float)


error_s = []
error_s = y_pred_st_1m - y_lab_st_1m

plt.hist(error_s, 100, linewidth=1.5, edgecolor='black', color='orange')
plt.xticks(np.arange(-1, 0.6, 0.1))
plt.xlim(-0.6, 0.6)
plt.title('Tuning model prediction error')
# plt.xlabel('Error')
plt.grid(True)
#plt.savefig('def_code/immagini/MLP/20_100x5/train_on_one_month_small/LSTM_tuning_model_error({}_epochs).png'.format(epochs_s))
plt.show()


plt.plot(y_pred_st_1m, color='orange', label="Predicted")
plt.plot(y_lab_st_1m, color="b", linestyle="dashed", linewidth=1, label="Real")
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlim(left=0, right=150)
plt.ylabel('Mean Air Temperature [°C]')
plt.xlabel('Time [h]')
plt.title("Tuning: real VS predicted temperature", size=15)
plt.legend()
#plt.savefig('def_code/immagini/MLP/20_100x5/train_on_one_month_small/LSTM_tuning_real_VS_predicted_temperature({}_epochs).png'.format(epochs_s))
plt.show()


MAPE = mean_absolute_percentage_error(y_lab_st_1m, y_pred_st_1m)
MSE = mean_squared_error(y_lab_st_1m, y_pred_st_1m)
R2 = r2_score(y_lab_st_1m, y_pred_st_1m)

print('MAPE:%0.5f%%'%MAPE)
print('MSE:', MSE.item())
print('R2:', R2.item())


plt.scatter(y_lab_st_1m, y_pred_st_1m,  color='k', edgecolor= 'white', linewidth=1) # ,alpha=0.1
plt.text(23.2, 28.2, 'MAPE: {:.3f}'.format(MAPE), fontsize=15, bbox=dict(facecolor='red', alpha=0.5))
plt.text(23.2, 29.2, 'MSE: {:.3f}'.format(MSE), fontsize=15, bbox=dict(facecolor='green', alpha=0.5))
plt.plot([23, 27, 30], [23, 27, 30], color='red')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlabel('Real Temperature [°C]')
plt.ylabel('Predicted Temperature [°C]')
plt.title("Tuning prediction distribution", size=15)
#plt.savefig('def_code/immagini/MLP/20_100x5/train_on_one_month_small/LSTM_tuning_prediction_distribution({}_epochs).png'.format(epochs_s))
plt.show()






