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

# from new_datasets_LSTM import read_csv

# Small_office
small_office_100 = read_csv(directory='small_office', file_csv='Small_office_100.csv')
small_office_random = read_csv(directory='small_office', file_csv='Small_office_random.csv')
small_office_105 = read_csv(directory='small_office', file_csv='Small_office_105.csv')
small_office_100_random_potenza_60_perc = read_csv(directory='Small_office', file_csv='Small_office_100_random_potenza_60_perc.csv')

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

small_office = pd.DataFrame()
small_office = concat_datasets(list=[small_office_100, small_office_105, small_office_random, small_office_100_random_potenza_60_perc], columns=columns)#  name=small_office

maxT_s = small_office['Mean air Temperature [°C]'].max()
minT_s = small_office['Mean air Temperature [°C]'].min()

def normalization(df):
    df = (df - df.min()) / (df.max() - df.min())
    return df

small_office = normalization(small_office)
"""
def binary_plot(df, col, title):
    plt.plot(df[col])
    plt.xlim(0, 500)
    plt.title('Small office occupation (binary plot)', size=15)
    # plt.savefig('def_code/immagini/binary_occupation/+'+title+'.png')
    plt.show()

binary_plot(small_office, 'Total People', title='small_office_binary_plot')
"""

# ______________________________________Datasets_preprocessing___________________________________________________________
# shifting_period = 1
period = 1
l_train = int(0.5 * len(small_office))
l_val = int(l_train+2928)# training length
# from new_datasets_LSTM import create_data

def create_data(df, col_name):
    train_mx = pd.DataFrame(df[:l_train])
    val_mx = pd.DataFrame(df[l_train:l_val])
    test_mx = pd.DataFrame(df[l_val:])
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

train_total_s, val_total_s, test_total_s = create_data(df=small_office, col_name='Mean air Temperature [°C]')
train_total_s, val_total_s, test_total_s = train_total_s.to_numpy(), val_total_s.to_numpy(), test_total_s.to_numpy()


# _____________________________________Split_the_x_and_y_datasets________________________
n_steps = 48

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

# from new_datasets_LSTM import split_sequences

# Split small office
train_total_sX, train_total_sY = split_sequences(sequences=train_total_s, n_steps=n_steps)
val_total_sX, val_total_sY = split_sequences(sequences=val_total_s, n_steps=n_steps)
test_total_sX, test_total_sY = split_sequences(sequences=test_total_s, n_steps=n_steps)

# Convert small office to tensors
train_total_sX = torch.from_numpy(train_total_sX)
train_total_sY = torch.from_numpy(train_total_sY)
val_total_sX = torch.from_numpy(val_total_sX)
val_total_sY = torch.from_numpy(val_total_sY)
test_total_sX = torch.from_numpy(test_total_sX)
test_total_sY = torch.from_numpy(test_total_sY)

print(type(train_total_sX), train_total_sX.shape)
print(type(train_total_sY), train_total_sY.shape)
print(type(val_total_sX), val_total_sX.shape)
print(type(val_total_sY), val_total_sY.shape)
print(type(test_total_sX), test_total_sX.shape)
print(type(test_total_sY), test_total_sY.shape)



# ______________________________________________LSTM_Structure__________________________________________________________
# HYPER PARAMETERS
lookback = 48
# train_episodes = 25
lr = 0.008 #0.005 #0.009
num_layers = 5
num_hidden = 15
batch_size = 100

train_batch_size = 500
train_data_total_s = TensorDataset(train_total_sX, train_total_sY)
train_dl_total_s = DataLoader(train_data_total_s, batch_size=train_batch_size, shuffle=True, drop_last=True)

val_batch_size = 300
val_data_total_s = TensorDataset(val_total_sX, val_total_sY)
val_dl_total_s = DataLoader(val_data_total_s, batch_size=val_batch_size, shuffle=True, drop_last=True)

# Structure
class MV_LSTM(torch.nn.Module):
    def __init__(self, n_features, seq_length, drop_prob=0.2):
        super(MV_LSTM, self).__init__()
        self.seq_len = seq_length
        self.n_hidden = num_hidden # number of hidden states
        self.n_layers = num_layers # number of LSTM layers (stacked)
        self.l_lstm = torch.nn.LSTM(input_size = n_features,
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers,
                                 batch_first = True)
        # self.dropout = torch.nn.Dropout(drop_prob)
        # according to pytorch docs LSTM output isn(batch_size,seq_len, num_directions * hidden_size) when considering batch_first = True
        self.l_linear = torch.nn.Sequential(
            nn.Linear(self.n_hidden, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
    def forward(self, x, h):
        batch_size, seq_len, _ = x.size()
        # hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        # cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        lstm_out, h = self.l_lstm(x, h)
        # h_out_numpy = h[0].detach().numpy() # se n layer = 1 all'ora h_out_numpy è ugugla a out_numpy2
        # out_numpy = lstm_out.detach().numpy()
        out = lstm_out[:, -1, :]
        # out_numpy2 = out.detach().numpy()#many to one, I take only the last output vector, for each Batch
        out_linear_transf = self.l_linear(out)
        return out_linear_transf, h

    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        hidden = (hidden_state, cell_state) #HIDDEN is defined as a TUPLE
        return hidden


# Create NN
#generalize the number of features and the number of timesteps by linking them to the preprocessing
n_features = train_total_sX.shape[2]
n_timesteps = lookback

#initialize the network,criterion and optimizer
lstm = MV_LSTM(n_features, n_timesteps)
criterion_total_s = torch.nn.MSELoss() # reduction='sum' created huge loss value
optimizer_total_s = torch.optim.Adam(lstm.parameters(), lr=lr)

# from new_datasets_LSTM import train_model

def train_model(model, epochs, train_dl, val_dl, optimizer, criterion, mode=''):
    # START THE TRAINING PROCESS
    model.train()
    # initialize the training loss and the validation loss
    TRAIN_LOSS = []
    VAL_LOSS = []

    for t in range(epochs):

        # TRAINING LOOP
        loss = []
        h = model.init_hidden(train_batch_size)  #hidden state is initialized at each epoch
        for x, label in train_dl:
            h = model.init_hidden(train_batch_size) #since the batch is big enough, a stateless mode is used (also considering the possibility to shuffle the training examples, which increase the generalization ability of the network)
            h = tuple([each.data for each in h])
            output, h = model(x.float(), h)
            label = label.unsqueeze(1) #utilizzo .unsqueeze per non avere problemi di dimensioni
            loss_c = criterion(output, label.float())
            optimizer.zero_grad()
            loss_c.backward()
            optimizer.step()
            loss.append(loss_c.item())
        TRAIN_LOSS.append(np.sum(loss)/train_batch_size)
        if mode == 'tuning':
            lr_scheduler.step()
        # print("Epoch: %d, training loss: %1.5f" % (train_episodes, LOSS[-1]))

        # VALIDATION LOOP
        val_loss = []
        h = model.init_hidden(val_batch_size)
        for inputs, labels in val_dl:
            h = tuple([each.data for each in h])
            val_output, h = model(inputs.float(), h)
            val_labels = labels.unsqueeze(1)
            val_loss_c = criterion(val_output, val_labels.float())
            val_loss.append(val_loss_c.item())
        # VAL_LOSS.append(val_loss.item())
        VAL_LOSS.append(np.sum(val_loss)/val_batch_size)
        print('Epoch : ', t, 'Training Loss : ', TRAIN_LOSS[-1], 'Validation Loss :', VAL_LOSS[-1])

    return TRAIN_LOSS, VAL_LOSS

epochs_s= 100
train_loss_total_s, val_loss_total_s = train_model(lstm, epochs=epochs_s, train_dl=train_dl_total_s, val_dl=val_dl_total_s, optimizer=optimizer_total_s, criterion=criterion_total_s)


# Plot to verify validation and train loss, in order to avoid underfitting and overfitting
plt.plot(train_loss_total_s, '--',color='r', linewidth = 1, label = 'Train Loss')
plt.plot(val_loss_total_s, color='b', linewidth = 1, label = 'Validation Loss')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
plt.xticks(np.arange(0, int(epochs_s), 20))
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.title("Training VS Validation loss", size=15)
plt.legend()
#plt.savefig('def_code/immagini/total_small_office/test_random_60_perc/LSTM_Train_VS_Val_LOSS({}_epochs).png'.format(epochs_s))
plt.show()

# _____________________________________________________SAVE THE MODEL ____________________________________________________

"""
torch.save(lstm.state_dict(), 'def_code/lstm_total_small_office.pth')
model = MV_LSTM(n_features, n_timesteps)
model.load_state_dict(torch.load('def_code/lstm_total_small_office.pth'))
model.eval()
"""


# ______________________________________________ 1h PREDICTION TESTING _____________________________________________

test_data_total_s = TensorDataset(test_total_sX, test_total_sY)
test_dl_total_s = DataLoader(test_data_total_s, shuffle=False, batch_size=val_batch_size, drop_last=True)
test_losses = []

# h = lstm.init_hidden(val_batch_size)

def test_model(model, test_dl, maxT, minT):
    h = model.init_hidden(val_batch_size)
    model.eval()
    y_pred = []
    y_lab = []

    for inputs, labels in test_dl:
        h = tuple([each.data for each in h])
        test_output, h = model(inputs.float(), h)
        labels = labels.unsqueeze(1)
        test_output = test_output.detach().numpy()

        #RESCALE OUTPUT
        test_output = np.reshape(test_output, (-1, 1))
        test_output = minT + test_output*(maxT-minT)

        # labels = labels.item()
        labels = labels.detach().numpy()
        labels = np.reshape(labels, (-1, 1))
        #RESCALE LABELS
        labels = minT + labels*(maxT-minT)
        y_pred.append(test_output)
        y_lab.append(labels)
    return y_pred, y_lab

# from new_datasets_LSTM import test_model

y_pred_s, y_lab_s = test_model(model, test_dl_total_s, maxT_s, minT_s)

flatten = lambda l: [item for sublist in l for item in sublist]
y_pred_s = flatten(y_pred_s)
y_lab_s = flatten(y_lab_s)
y_pred_s = np.array(y_pred_s, dtype=float)
y_lab_s = np.array(y_lab_s, dtype=float)


error_s = []
error_s = y_pred_s - y_lab_s

plt.hist(error_s, 100, linewidth=1.5, edgecolor='black', color='orange')
plt.xticks(np.arange(-0.4, 0.4, 0.1))
plt.xlim(-0.4, 0.4)
plt.title('LSTM model prediction error')
# plt.xlabel('Error')
plt.grid(True)
#plt.savefig('def_code/immagini/total_small_office/test_random_60_perc/LSTM_model_error({}_epochs).png'.format(epochs_s))
plt.show()


plt.plot(y_pred_s, color='orange', label="Predicted")
plt.plot(y_lab_s, color="b", linestyle="dashed", linewidth=1, label="Real")
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlim(left=0,right=800)
plt.ylabel('Mean Air Temperature [°C]')
plt.xlabel('Time [h]')
plt.title("Real VS predicted temperature", size=15)
plt.legend()
#plt.savefig('def_code/immagini/total_small_office/test_random_60_perc/LSTM_real_VS_predicted_temperature({}_epochs).png'.format(epochs_s))
plt.show()


# METRICS
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

MAPE = mean_absolute_percentage_error(y_lab_s, y_pred_s)
RMSE = mean_squared_error(y_lab_s,y_pred_s)**0.5
R2 = r2_score(y_lab_s,y_pred_s)

print('MAPE:%0.5f%%'%MAPE)
print('RMSE:', RMSE.item())
print('R2:', R2.item())


plt.scatter(y_lab_s, y_pred_s,  color='k', edgecolor= 'white', linewidth=1,alpha=0.1)
plt.text(24.5, 29.2, 'MAPE: {:.3f}'.format(MAPE), fontsize=15, bbox=dict(facecolor='red', alpha=0.5))
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlabel('Real Temperature [°C]')
plt.ylabel('Predicted Temperature [°C]')
plt.title("Prediction distribution", size=15)
# plt.savefig('def_code/immagini/total_small_office/test_random_60_perc/LSTM_prediction_distribution({}_epochs).png'.format(epochs_s))
plt.show()
