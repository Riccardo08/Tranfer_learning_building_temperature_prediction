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
restaurant = concat_datasets(list=[restaurant_100, restaurant_100_potenza_random_60_percento, restaurant_dataset_validation, restaurant_random], columns=columns)
retail = concat_datasets(list=[retail_100, retail_100_potenza_random_60_percento, retail_105, retail_random], columns=columns)

maxT_m = medium_office['Mean air Temperature [°C]'].max()
minT_m = medium_office['Mean air Temperature [°C]'].min()
maxT_s = small_office['Mean air Temperature [°C]'].max()
minT_s = small_office['Mean air Temperature [°C]'].min()

def normalization(df):
    df = (df - df.min()) / (df.max() - df.min())
    # df = (df[col_names] - df[col_names].min()) / (df[col_names].max() - df[col_names].min())
    return df

medium_office = normalization(medium_office)
small_office = normalization(small_office)
restaurant = normalization(restaurant)
retail = normalization(retail)

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




#_________________________________________________CNN_MODEL_____________________________________________________
class CNN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CNN, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = 2
        self.pad = 0
        self.dil = 1
        self.str = 1
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channel, out_channels=self.out_channel, kernel_size=self.kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(self.out_channel, 1, self.kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(2)
            #nn.Conv1d(1, 1, self.kernel_size),
            #nn.ReLU(),
            #nn.MaxPool1d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=59, out_features=40),  # need to change the input (20).
            nn.ReLU(),
            nn.Linear(40, 1)
            #nn.ReLU(),
            #nn.Linear(20, 1)
            # nn.ReLU(),
            # nn.Linear(20, 10),
            # nn.ReLU(),
            #nn.Linear(10, 1)
        )
    def forward(self, x):
        x = self.conv(x)
        # x = self.pool2(F.relu(self.conv3(x)))
        # x = nn.flatten(x, 1, -1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


#______________________________________________Define_PARAMETERS______________________________________________
learning_rate = 0.008
in_channels = 1
out_channels = 2
# batch_size = 100
train_batch_size = 500
val_batch_size = 150
test_batch_size = 200

# Define model, criterion and optimizer:
cnn = CNN(in_channels, out_channels)
criterion_m = torch.nn.MSELoss()
optimizer_m = torch.optim.SGD(cnn.parameters(), lr=learning_rate)


# Dataloaders:
train_data_m = TensorDataset(train_mX, train_mY)
train_dl_m = DataLoader(train_data_m, batch_size=train_batch_size, shuffle=True, drop_last=True)

val_data_m = TensorDataset(val_mX, val_mY)
val_dl_m = DataLoader(val_data_m, batch_size=val_batch_size, shuffle=True, drop_last=True)


def train_model(model, criterion, optimizer, num_epochs, train_dl, val_dl, mode=''):
    since = time.time()

    # best_model_wts = copy.deepcopy(model.state_dict())  # 'state_dict' mappa ogni layer col suo tensore dei parametri
    # best_acc = 0.0
    model.train()
    # initialize the training loss and the validation loss
    TRAIN_LOSS = []
    VAL_LOSS = []
    val_output_list = []
    val_labels_list = []
    running_corrects = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 20)
        #_____________________________________TRAINING_LOOP____________________________________________
        loss = []
        for x, label in train_dl:
            x = torch.reshape(x.float(), (train_batch_size, in_channels, x.shape[1] * x.shape[2]))  # 100, 1, 48*6 --> (100, 1, 288)
            output = model(x)
            label = label.unsqueeze(1)
            loss_c = criterion(output, label.float())
            optimizer.zero_grad()
            loss_c.backward()
            optimizer.step()
            loss.append(loss_c.item())
        TRAIN_LOSS.append(np.sum(loss) / train_batch_size)
        if mode == 'tuning':
            lr_scheduler.step()

        #________________________________________VALIDATION LOOP_______________________________________
        val_loss = []
        for inputs, labels in val_dl:
            inputs = torch.reshape(inputs.float(), (val_batch_size, in_channels, inputs.shape[1] * inputs.shape[2]))
            val_output = model(inputs.float())
            val_labels = labels.unsqueeze(1)
            val_loss_c = criterion(val_output, val_labels.float())
            # VAL_LOSS.append(val_loss.item())
            val_loss.append(val_loss_c.item())
            val_output_list.append(val_output)
            val_labels_list.append(val_labels)
        VAL_LOSS.append(np.sum(val_loss) / val_batch_size)

        print('Epoch : ', epoch, 'Training Loss : ', TRAIN_LOSS[-1], 'Validation Loss :', VAL_LOSS[-1])

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return TRAIN_LOSS, VAL_LOSS

epochs_m = 350
train_loss_m, val_loss_m = train_model(cnn, criterion_m, optimizer_m, num_epochs=epochs_m, train_dl=train_dl_m, val_dl=val_dl_m)

#Plot to verify validation and train loss, in order to avoid underfitting and overfitting
plt.plot(train_loss_m,'--', color='r', linewidth = 1, label = 'Train Loss')
plt.plot(val_loss_m, color='b', linewidth = 1, label = 'Validation Loss')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
# plt.xticks(np.arange(0, epochs, 1))
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.title("Training VS Validation loss", size=15)
plt.legend()
#plt.savefig('def_code/immagini/CNN/CNN_Train_VS_Val_LOSS({}_epochs).png'.format(epochs_m))
plt.show()


#_________________________________________________TESTING_PHASE_______________________________________________

test_data_m = TensorDataset(test_mX, test_mY)
test_dl_m = DataLoader(test_data_m, batch_size=test_batch_size, drop_last=True)

def test_model(model, test_dl, maxT, minT):
    model.eval()
    test_losses = []
    y_pred = []
    y_lab = []

    for inputs, labels in test_dl:
        inputs = torch.reshape(inputs, (test_batch_size, in_channels, inputs.shape[1]*inputs.shape[2]))
        outputs = model(inputs.float())
        #outputs = outputs.detach().numpy()
        #outputs = np.reshape(outputs, (-1, 1))
        outputs = minT + outputs*(maxT-minT)

        labs = labels.unsqueeze(1)
        # labs = labs.float()
        #labels = labs.detach().numpy()
        #labs = np.reshape(labs, (-1, 1))
        labs = minT + labs*(maxT-minT)

        y_pred.append(outputs)
        y_lab.append(labs)
    return y_pred, y_lab


y_pred_m, y_lab_m = test_model(cnn, test_dl_m, maxT_m, minT_m)

flatten = lambda l: [item for sublist in l for item in sublist]
y_pred_m = flatten(y_pred_m)
y_lab_m = flatten(y_lab_m)
y_pred_m = np.array(y_pred_m, dtype=float)
y_lab_m = np.array(y_lab_m, dtype=float)
y_pred_m = np.reshape(y_pred_m, (-1, 1))
y_lab_m = np.reshape(y_lab_m, (-1, 1))

error_m = []
error_m = y_pred_m - y_lab_m

# Plot the error
plt.hist(error_m, 200, linewidth=1.5, edgecolor='black', color='orange')
plt.xticks(np.arange(-0.6, 0.6, 0.1))
plt.xlim(-0.6, 0.6)
plt.title('First model prediction error')
# plt.xlabel('Error')
plt.grid(True)
# plt.savefig('def_code/immagini/CNN/CNN_model_error.png')
plt.show()


plt.plot(y_pred_m, color='orange', label="Predicted")
plt.plot(y_lab_m, color="b", linestyle="dashed", linewidth=1, label="Real")
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlim(left=0, right=300)
plt.ylabel('Mean Air Temperature [°C]')
plt.xlabel('Time [h]')
plt.title("Real VS predicted temperature", size=15)
plt.legend()
# plt.savefig('def_code/immagini/CNN/CNN_real_VS_predicted_temperature({}_epochs).png'.format(epochs_m))
plt.show()


#______________________________METRICS_EVALUATION___________________________________________________
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

MAPE = mean_absolute_percentage_error(y_lab_m, y_pred_m)
RMSE=mean_squared_error(y_lab_m, y_pred_m)**0.5
R2 = r2_score(y_lab_m, y_pred_m)

print('MAPE:%0.5f%%'%MAPE)      # MAPE < 10% is Excellent, MAPE < 20% is Good.
print('RMSE:%0.5f'%RMSE.item()) # RMSE values between 0.2 and 0.5 shows that the model can relatively predict the data accurately.
print('R2:%0.5f'%R2.item())     # R-squared more than 0.75 is a very good value for showing the accuracy.

plt.scatter(y_lab_m, y_pred_m,  color='k', edgecolor= 'white', linewidth=1,alpha=0.1)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlabel('Real Temperature [°C]')
plt.ylabel('Predicted Temperature [°C]')
plt.title("Prediction distribution", size=15)
#plt.savefig('def_code/immagini/CNN/CNN_prediction_distribution({}_epochs).png'.format(epochs_m))
plt.show()


#_____________________________________________________DEFINE_TUNING_PHASE_______________________________________________
def freeze_params(model):
    for param_c in model.conv.parameters():
            param_c.requires_grad = True
    for param_fc in model.fc.parameters():
            param_fc.requires_grad = False
    return model

cnn_test = freeze_params(cnn)

print(cnn_test)
for i in cnn_test.conv.parameters():
    print(i)
for x in cnn_test.fc.parameters():
    print(x)

num_ftrs = cnn_test.fc[2].in_features

cnn_test.fc[2] = nn.Linear(num_ftrs, 30)
#cnn_test.fc[3] = nn.ReLU()
#cnn_test.fc[4] = nn.Linear(30, 1)
cnn_test.fc.add_module('3', nn.ReLU())
cnn_test.fc.add_module('4', nn.Linear(30, 1))

cnn_test.fc.add_module('9', nn.ReLU())
cnn_test.fc.add_module('10', nn.Linear(10, 1))
print(cnn_test)
# How to delete some layers from the model:
# cnn_test.fc = nn.Sequential(*[cnn_test.fc[i] for i in range(4, len(cnn_test.fc))])

criterion_ft = torch.nn.MSELoss()
# optimizer_ft = torch.optim.SGD(cnn_test.parameters(), lr=learning_rate)
optimizer_ft = optim.SGD(filter(lambda p: p.requires_grad, cnn_test.parameters()), lr=0.004)

# Decay LR (learning rate) by a factor of 0.1 every 7 epochs
lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


#__________________________________INCLUDE_NEW_DATASET__________________________________________________________________
# from new_dataset import train_mX_new, train_mY_new, val_mX_new, val_mY_new, test_mX_new, test_mY_new

#New Dataloaders
train_batch_size = 500
train_data_s = TensorDataset(train_sX, train_sY)
train_dl_s = DataLoader(train_data_s, batch_size=train_batch_size, shuffle=True, drop_last=True)

val_batch_size = 300
val_data_s = TensorDataset(val_sX, val_sY)
val_dl_s = DataLoader(val_data_s, batch_size=val_batch_size, shuffle=True, drop_last=True)

epochs_s = 150
#_TRAIN_THE_MODEL_________________________________________________________________________________
train_loss_s, val_loss_s = train_model(cnn_test, criterion_ft, optimizer_ft, epochs_s, train_dl=train_dl_s, val_dl=val_dl_s, mode='tuning')


#Plot to verify validation and train loss, in order to avoid underfitting and overfitting
plt.plot(train_loss_s,'--',color='r', linewidth = 1, label = 'Train Loss')
plt.plot(val_loss_s,color='b', linewidth = 1, label = 'Validation Loss')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
plt.ylim(bottom=0)
# plt.xticks(np.arange(0, epochs, 1))
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.title("Training VS Validation loss", size=15)
plt.legend()
#plt.savefig('def_code/immagini/CNN/tuning/CNN_tuning_Train_VS_Val_LOSS({}_epochs).png'.format(epochs_s))
plt.show()


#_________________________________________________TESTING_PHASE_______________________________________________
# loadedmodel.eval()
test_data_s = TensorDataset(test_sX, test_sY)
test_dl_s = DataLoader(test_data_s, batch_size=test_batch_size, shuffle=False, drop_last=True) # batch_size -> terza dimensione


y_pred_s, y_lab_s = test_model(cnn_test, test_dl_s, maxT_s, minT_s)

flatten = lambda l: [item for sublist in l for item in sublist]
y_pred_s = flatten(y_pred_s)
y_lab_s = flatten(y_lab_s)
y_pred_s = np.array(y_pred_s, dtype=float)
y_lab_s = np.array(y_lab_s, dtype=float)
y_pred_s = np.reshape(y_pred_s, (-1, 1))
y_lab_s = np.reshape(y_lab_s, (-1, 1))

error_s = []
error_s = y_pred_s - y_lab_s


# Plot the error
plt.hist(error_s, 200, linewidth=1.5, edgecolor='black', color='orange')
plt.xticks(np.arange(-0.6, 0.6, 0.1))
plt.xlim(-0.6, 0.6)
plt.title('CNN tuning model prediction error')
# plt.xlabel('Error')
plt.grid(True)
#plt.savefig('def_code/immagini/CNN/tuning/CNN_tuning_model_error({}_epochs).png'.format(epochs_s))
plt.show()


plt.plot(y_pred_s, color='orange', label="Predicted")
plt.plot(y_lab_s, color="b", linestyle="dashed", linewidth=1, label="Real")
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlim(left=4500, right=5000)
plt.ylabel('Mean Air Temperature [°C]')
plt.xlabel('Time [h]')
plt.title("Real VS predicted temperature", size=15)
plt.legend()
#plt.savefig('def_code/immagini/CNN/tuning/CNN_tuning_real_VS_predicted_temperature({}_epochs).png'.format(epochs_s))
plt.show()


#______________________________METRICS_EVALUATION___________________________________________________
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

MAPE = mean_absolute_percentage_error(y_lab_s, y_pred_s)
RMSE = mean_squared_error(y_lab_s, y_pred_s)**0.5
R2 = r2_score(y_lab_s, y_pred_s)

print('MAPE:%0.5f%%'%MAPE)      # MAPE < 10% is Excellent, MAPE < 20% is Good.
print('RMSE:%0.5f'%RMSE.item()) # RMSE values between 0.2 and 0.5 shows that the model can relatively predict the data accurately.
print('R2:%0.5f'%R2.item())     # R-squared more than 0.75 is a very good value for showing the accuracy.

plt.scatter(y_lab_s, y_pred_s,  color='k', edgecolor= 'white', linewidth=1,alpha=0.1)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlabel('Real Temperature [°C]')
plt.ylabel('Predicted Temperature [°C]')
plt.title("Prediction distribution", size=15)
#plt.savefig('def_code/immagini/CNN/tuning/CNN_tuning_prediction_distribution({}_epochs).png'.format(epochs_s))
plt.show()



