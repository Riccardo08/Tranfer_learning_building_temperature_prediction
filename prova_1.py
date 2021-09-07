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


plt.ion()   # interactive mode


df = pd.read_csv('ASHRAE90.1_OfficeSmall_STD2016_NewYork.csv')
df=df.iloc[288:,:]
Date = df['Date/Time'].str.split(' ', expand=True)
Date.rename(columns={0:'nullo',1:'date',2:'null', 3:'time'},inplace=True)
Date['time'] = Date['time'].replace(to_replace='24:00:00', value= '0:00:00')
data = Date['date']+' '+Date['time']
data = pd.to_datetime(data, format='%m/%d %H:%M:%S')

df['day']=data.apply(lambda x: x.day)
df['month']=data.apply(lambda x: x.month)
df['hour']=data.apply(lambda x: x.hour)
df['dn']=data.apply(lambda x: x.weekday())
df['data']=Date.date

df = df.reset_index(drop=True)

"""
# Shift training and testing input dataset
def shifting(col, period):
    new_col = col.shift(periods=period) # shifting
    new_col = new_col.reset_index(drop=True) # reset the index of the rows
    new_col = new_col.iloc[period:] # delete the Nan
    return new_col
"""

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


# create the list of input columns
col_names = ['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)',
             'Environment:Site Diffuse Solar Radiation Rate per Area [W/m2](TimeStep)',
             'CORE_ZN:Zone People Occupant Count [](TimeStep)',
             'PSZ-AC:1 HEAT PUMP DX COOLING COIL:Cooling Coil Total Cooling Rate [W](TimeStep)',
             'PSZ-AC:1 HEAT PUMP DX HEATING COIL:Heating Coil Heating Rate [W](TimeStep)',
             'CORE_ZN:Zone Mean Air Temperature [C](TimeStep)']

multi_data = (df[col_names]-df[col_names].min())/(df[col_names].max()-df[col_names].min())
multi_norm = multi_data

#_____________________________________________Normalization____________________________________
l_train = int(0.8 * len(df))
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
train_mx, test_mx, val_mx = multi_shift(multi_norm, col_name='CORE_ZN:Zone Mean Air Temperature [C](TimeStep)')
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



#_____________________________________Split_the_x_and_y_datasets________________________
n_steps = 48
train_mX, train_mY = split_sequences(train_m, n_steps=n_steps)
val_mX, val_mY = split_sequences(val_m, n_steps=n_steps)
test_mX, test_mY = split_sequences(test_m, n_steps=n_steps)

# Convert to tensors
train_mX = torch.from_numpy(train_mX)
train_mY = torch.from_numpy(train_mY)
val_mX = torch.from_numpy(val_mX)
val_mY = torch.from_numpy(val_mY)
test_mX = torch.from_numpy(test_mX)
test_mY = torch.from_numpy(test_mY)

print(type(train_mX), train_mX.shape)
print(type(train_mY), train_mY.shape)
print(type(val_mX), val_mX.shape)
print(type(val_mY), val_mY.shape)
print(type(test_mX), test_mX.shape)
print(type(test_mY), test_mY.shape)

"""
#_____________________________1D-CNN_(SLOW)___________________________________
class CNN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CNN, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = 2
        self.pad = 0
        self.dil = 1
        self.str = 1
        self.conv1 = nn.Conv1d(in_channels=self.in_channel, out_channels=self.out_channel, kernel_size=self.kernel_size, padding=self.pad, dilation=self.dil, stride=self.str)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(self.out_channel, 1, self.kernel_size)
        self.fc1 = nn.Linear(7100, 6000) # need to change the input (20).
        self.fc2 = nn.Linear(6000, 4500)
        self.fc3 = nn.Linear(4500, 2500)
        self.fc4 = nn.Linear(2500, 1000)
        self.fc5 = nn.Linear(1000, 700)
        self.fc6 = nn.Linear(700, 150)
        self.fc7 = nn.Linear(150, 80)
        self.fc8 = nn.Linear(80, 20)
        self.fc9 = nn.Linear(20, 1)



    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 7100)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = self.fc9(x)
        return x
"""


#_____________________________1D-CNN_(FAST)___________________________________
class CNN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CNN, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = 3
        self.pad = 0
        self.dil = 1
        self.str = 3
        self.conv1 = nn.Conv1d(in_channels=self.in_channel, out_channels=self.out_channel, kernel_size=self.kernel_size)
        self.maxpool = nn.MaxPool1d(kernel_size=self.kernel_size, stride=self.str)
        self.conv2 = nn.Conv1d(self.out_channel, 5, self.kernel_size)
        #self.pool2 = nn.MaxPool1d(2)
        # self.conv3 = nn.Conv1d(1, 1, self.kernel_size)
        self.fc1 = nn.Linear(5*self.kernel_size, 10) # need to change the input (20).
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(5, 1)
        # self.fc4 = nn.Linear(20, 1)

        def forward(self, x):
            x = self.maxpool(F.relu(self.conv1(x)))
            x = self.maxpool(x)

            x = self.maxpool(F.relu(self.conv2(x)))
            x = self.maxpool(x)

            x = x.view(-1, 5*self.kernel_size)
            x = x.flatten(1)  # flatten the tensor starting at dimension 1

            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)


"""
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)

        x = F.relu(self.conv2(x))
        x = self.maxpool(x)

        x = x.flatten(1)  # flatten the tensor starting at dimension 1

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
"""
"""
    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        #x = self.maxpool(F.relu(self.conv3(x)))
        x = x.view(-1, 71)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = self.fc4(x)
        return x
"""



#____________________________Define_PARAMETERS______________________________________________
epochs = 200
learning_rate = 0.009
# batch_size = 100
train_batch_size = 500
val_batch_size = 200
test_batch_size = 300
in_channels = 2
out_channels = 18


# Define model, criterion and optimizer:
cnn_model = CNN(in_channels, out_channels)
criterion = torch.nn.MSELoss() # reduction='sum' created huge loss value
optimizer = torch.optim.SGD(cnn_model.parameters(), lr=learning_rate)

# Dataloaders:
train_data = TensorDataset(train_mX, train_mY)
train_dl = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, drop_last=True)

val_data = TensorDataset(val_mX, val_mY)
val_dl = DataLoader(val_data, batch_size=val_batch_size, shuffle=True, drop_last=True)

test_data = TensorDataset(test_mX, test_mY)
test_dl = DataLoader(test_data, batch_size=test_batch_size, drop_last=True)


#initialize the training loss and the validation loss
LOSS = []
VAL_LOSS = []
val_output_list = []
val_labels_list = []


#START THE TRAINING PROCESS
cnn_model.train()

for t in range(epochs):
    # h = cnn_model.init_hidden(batch_size)  #hidden state is initialized at each epoch
    loss = []
    for x, label in train_dl:
        # h = cnn_model.init_hidden(batch_size)                                     # since the batch is big enough, a stateless mode is used (also considering the possibility to shuffle the training examples, which increase the generalization ability of the network)
        # h = tuple([each.data for each in h])
        x = torch.reshape(x.float(), (train_batch_size, in_channels, x.shape[1]*x.shape[2]))  # batch_size, in_channles, 48*6 --> (batch_size, in_channels, 288)
        output = cnn_model(x)
        label = label.unsqueeze(1)
        loss_c = criterion(output, label.float())
        optimizer.zero_grad()
        loss_c.backward()
        optimizer.step()
        loss.append(loss_c.item())
    LOSS.append(np.sum(loss)/train_batch_size)
    #LOSS.append(loss)
    # print("Epoch: %d, training loss: %1.5f" % (train_episodes, LOSS[-1]))
    # print('Epoch : ', t, 'Training Loss : ', LOSS[-1])


    # VALIDATION LOOP
    val_loss =[]
    # h = mv_net.init_hidden(batch_size)
    for inputs, labels in val_dl:
        # h = tuple([each.data for each in h])
        inputs = torch.reshape(inputs.float(), (val_batch_size, in_channels, inputs.shape[1]*inputs.shape[2]))
        val_output = cnn_model(inputs.float())
        val_labels = labels.unsqueeze(1)
        val_loss_c = criterion(val_output, val_labels.float())
        # VAL_LOSS.append(val_loss.item())
        val_loss.append(val_loss_c.item())
        val_output_list.append(val_output)
        val_labels_list.append(val_labels)
    VAL_LOSS.append(np.sum(val_loss)/val_batch_size)
    print('Epoch : ', t, 'Training Loss : ', LOSS[-1], 'Validation Loss :', VAL_LOSS[-1])


"""
flatten = lambda l: [item for sublist in l for item in sublist]
val_output_list = flatten(val_output_list)
val_labels_list = flatten(val_labels_list)
val_output_list = np.array(val_output_list, dtype=float)
val_labels_list = np.array(val_labels_list, dtype=float)
val_output_list = np.reshape(val_output_list, (-1, 1))
val_labels_list = np.reshape(val_labels_list, (-1, 1))
"""
# x = torch.rand(100, 1, 288)
# print(cnn_model(x).shape)

#Plot to verify validation and train loss, in order to avoid underfitting and overfitting
plt.plot(LOSS,'--',color='r', linewidth = 1, label = 'Train Loss')
plt.plot(VAL_LOSS,color='b', linewidth = 1, label = 'Validation Loss')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
# plt.xticks(np.arange(0, epochs, 1))
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.title("Training VS Validation loss", size=15)
plt.legend()
# plt.savefig('immagini_LSTM/I_LSTM_Train_VS_Val_LOSS(10_epochs).png')
plt.show()



#_____________________________________SAVE_THE_MODEL_____________________________
"""
torch.save(cnn_model.state_dict(), "cnn_model_weight.pth")
loadedmodel = CNN(1, 2)
loadedmodel.load_state_dict(torch.load("cnn_model_weight.pth"))
"""


#___________________________TESTING_PHASE________________________________________
# loadedmodel.eval()
cnn_model.eval()
test_losses = []
ypred=[]
ylab=[]

for inputs, labels in test_dl:
    inputs = torch.reshape(inputs, (test_batch_size, 2, inputs.shape[1]*inputs.shape[2]))
    outputs = cnn_model(inputs.float())
    #outputs = outputs.detach().numpy()
    #outputs = np.reshape(outputs, (-1, 1))
    outputs = minT + outputs*(maxT-minT)

    labs = labels.unsqueeze(1)
    # labs = labs.float()
    #labels = labs.detach().numpy()
    #labs = np.reshape(labs, (-1, 1))
    labs = minT + labs*(maxT-minT)

    ypred.append(outputs)
    ylab.append(labs)


flatten = lambda l: [item for sublist in l for item in sublist]
ypred = flatten(ypred)
ylab = flatten(ylab)
ypred = np.array(ypred, dtype=float)
ylab = np.array(ylab, dtype=float)
ypred = np.reshape(ypred, (-1, 1))
ylab = np.reshape(ylab, (-1, 1))

error = []
error = ypred - ylab

"""
#________________________________VERIFICA: ylab == test_mY_____________________________________________
test_mY_prova = test_mY.numpy()
for x in range(0, len(test_mY_prova)):
    test_mY_prova[x] = minT + test_mY_prova[x]*(maxT - minT)

test_mY_prova = np.reshape(test_mY_prova, (-1, 1))
test_mY_prova = test_mY_prova[:10400]

n=0
for x in range(0, len(test_mY_prova)):      # --------> n rimane = 0 => i die vettori sono gli stessi
    if test_mY_prova[x]==test_mY[x]:
        n += 1

#______________________________________________________________________________________________________
"""

# Plot the error
plt.hist(error, 200, linewidth=1.5, edgecolor='black', color='orange')
plt.xticks(np.arange(-0.6, 0.6, 0.1))
plt.xlim(-0.6, 0.6)
plt.title('First model prediction error')
# plt.xlabel('Error')
plt.grid(True)
# plt.savefig('immagini_LSTM/first_model_error.png')
plt.show()


plt.plot(ypred, color='orange', label="Predicted")
plt.plot(ylab, color="b", linestyle="dashed", linewidth=1, label="Real")
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlim(left=4500, right=5000)
plt.ylabel('Mean Air Temperature [°C]')
plt.xlabel('Time [h]')
plt.title("Real VS predicted temperature", size=15)
plt.legend()
# plt.savefig('immagini_LSTM/final_LSTM_real_VS_predicted_temperature(10_neurons).png')
plt.show()


#______________________________METRICS_EVALUATION___________________________________________________
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

MAPE = mean_absolute_percentage_error(ylab, ypred)
RMSE=mean_squared_error(ylab,ypred)**0.5
R2 = r2_score(ylab, ypred)

print('MAPE:%0.5f%%'%MAPE)      # MAPE < 10% is Excellent, MAPE < 20% is Good.
print('RMSE:%0.5f'%RMSE.item()) # RMSE values between 0.2 and 0.5 shows that the model can relatively predict the data accurately.
print('R2:%0.5f'%R2.item())     # R-squared more than 0.75 is a very good value for showing the accuracy.

plt.scatter(ylab, ypred,  color='k', edgecolor= 'white', linewidth=1,alpha=0.1)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlabel('Real Temperature [°C]')
plt.ylabel('Predicted Temperature [°C]')
plt.title("Prediction distribution", size=15)
# plt.savefig('immagini_LSTM/final_LSTM_prediction_distribution(10_neurons).png')
plt.show()



