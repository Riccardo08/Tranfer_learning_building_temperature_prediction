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
"""
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
"""

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
        #else:
        #    name['Total People'][i] = 1.0
    return name

medium_office = pd.DataFrame()
# small_office = pd.DataFrame()
# restaurant = pd.DataFrame()
# retail = pd.DataFrame()

medium_office = concat_datasets(list=[medium_office_2_100, medium_office_2_dataset_validation, medium_office_2_random_2,medium_office_2_100_random_60_perc], columns=columns) # name=medium_office
# small_office = concat_datasets(list=[small_office_100, small_office_105, small_office_random, small_office_100_random_potenza_60_perc], columns=columns)#  name=small_office
# restaurant = concat_datasets(list=[restaurant_100, restaurant_100_potenza_random_60_percento, restaurant_dataset_validation, restaurant_random], columns=columns)
# retail = concat_datasets(list=[retail_100, retail_100_potenza_random_60_percento, retail_105, retail_random], columns=columns)


# ___________________________________________________Normalization______________________________________________________

maxT_m = medium_office['Mean air Temperature [°C]'].max()
minT_m = medium_office['Mean air Temperature [°C]'].min()
# maxT_s = small_office['Mean air Temperature [°C]'].max()
# minT_s = small_office['Mean air Temperature [°C]'].min()

def normalization(df):
    df = (df - df.min()) / (df.max() - df.min())
    return df

medium_office = normalization(medium_office)
# small_office = normalization(small_office)
# restaurant = normalization(restaurant)
# retail = normalization(retail)

# __________________________________________________PLOTS________________________________________________________________

# LINE
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

medium_office['Environment:Site Day Type Index [](Hourly)'] = round(medium_office['Environment:Site Day Type Index [](Hourly)'], 2)
# small_office['Environment:Site Day Type Index [](Hourly)'] = round(small_office['Environment:Site Day Type Index [](Hourly)'], 2)
# restaurant['Environment:Site Day Type Index [](Hourly)'] = round(restaurant['Environment:Site Day Type Index [](Hourly)'], 2)
# retail['Environment:Site Day Type Index [](Hourly)'] = round(retail['Environment:Site Day Type Index [](Hourly)'], 2)
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
    fig.suptitle(dataset_name + ': ' + column, size=50)
    plt.yticks(np.arange(0, 1 ,0.1))
    plt.show()
    if column == 'Mean air Temperature [°C]':
        fig.savefig('def_code/immagini/mean_air_temperature/{}_{}.png'.format(dataset_name, column))
    if column == 'Total Cooling Rate [W]':
        fig.savefig('def_code/immagini/total_cooling_rate/{}_{}.png'.format(dataset_name, column))


# Plot mean air temperature
visualization(medium_office, 'Mean air Temperature [°C]', dataset_name='Medium office')
visualization(small_office, 'Mean air Temperature [°C]', dataset_name='Small office')
visualization(restaurant, 'Mean air Temperature [°C]', dataset_name='Restaurant')
visualization(retail, 'Mean air Temperature [°C]', dataset_name='Retail')


# Plot the total cooling rate
visualization(medium_office, 'Total Cooling Rate [W]', dataset_name='Medium office')
visualization(small_office, 'Total Cooling Rate [W]', dataset_name='Small office')
visualization(restaurant, 'Total Cooling Rate [W]', dataset_name='Restaurant')
visualization(retail, 'Total Cooling Rate [W]', dataset_name='Retail')


# BOXPLOT
def bx(df_list, column, by, type):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    df_list[0].boxplot(column=column, by=by, ax=axes[0, 0])
    axes[0, 0].title.set_text('Medium office')
    axes[0, 0].set_xlabel('Day index')
    # axes[0, 0].set_xticks(np.arange(0, 6, 1))

    df_list[1].boxplot(column=column, by=by, ax=axes[0, 1])
    axes[0, 1].title.set_text('Small office')
    axes[0, 1].set_xlabel('Day index')

    df_list[2].boxplot(column=column, by=by, ax=axes[1, 0])
    axes[1, 0].title.set_text('Restaurant')
    axes[1, 0].set_xlabel('Day index')

    df_list[3].boxplot(column=column, by=by, ax=axes[1, 1])
    axes[1, 1].title.set_text('Retail')
    axes[1, 1].set_xlabel('Day index')
    plt.suptitle(column, size=16)

    if column == 'Mean air Temperature [°C]':
        fig.savefig('def_code/immagini/mean_air_temperature/{}_{}.png'.format(type, column))
    if column == 'Total Cooling Rate [W]':
        fig.savefig('def_code/immagini/total_cooling_rate/{}_{}.png'.format(type, column))

    plt.show()


column = 'Mean air Temperature [°C]'
by = 'Environment:Site Day Type Index [](Hourly)'
bx(df_list=[medium_office, small_office, restaurant, retail], column=column, by=by, type='Boxplot')

column = 'Total Cooling Rate [W]'
bx(df_list=[medium_office, small_office, restaurant, retail], column=column, by=by, type='Boxplot')


# SINGLE BOXPLOTS
def view_boxplot(df, column, by, title):
    df.boxplot(by=by, column=column, grid=False)
    plt.title(title)
    plt.suptitle('')
    plt.xlabel('Day index')
    if column == 'Mean air Temperature [°C]':
        plt.yticks(np.arange(0, 1, 0.1))
    plt.show()

# BOXPLOT: Mean air temperature
column = 'Mean air Temperature [°C]'
by = 'Environment:Site Day Type Index [](Hourly)'
view_boxplot(medium_office, column=column, by=by, title='Medium office: ' + column)
view_boxplot(small_office, column=column, by=by, title='Small office: ' + column)
view_boxplot(restaurant, column=column, by=by, title='Restaurant: ' + column)
view_boxplot(retail, column=column, by=by, title='Retail: ' + column)

# BOXPLOT: Total cooling rate
column = 'Total Cooling Rate [W]'
by = 'Environment:Site Day Type Index [](Hourly)'
view_boxplot(medium_office, column=column, by=by, title='Medium office: ' + column)
view_boxplot(small_office, column=column, by=by, title='Small office: ' + column)
view_boxplot(restaurant, column=column, by=by, title='Restaurant: ' + column)
view_boxplot(retail, column=column, by=by, title='Retail: ' + column)


# BINARY PLOT - OCCUPATION
def binary_plot(df, col, title):
    plt.plot(df[col])
    plt.xlim(0, 500)
    plt.title('Medium office occupation (binary plot)', size=15)
    # plt.savefig('def_code/immagini/binary_occupation/+'+title+'.png')
    plt.show()


binary_plot(medium_office, 'Total People', title='medium_office_binary_plot')
binary_plot(small_office, 'Total People', title='medium_office_binary_plot')

# FREQUENCY - TOTAL COOLING RATE
weekday = pd.DataFrame({'medium': medium_office['Total Cooling Rate [W]'].groupby(
    medium_office['Environment:Site Day Type Index [](Hourly)']).mean(),
                        'small': small_office['Total Cooling Rate [W]'].groupby(
                            small_office['Environment:Site Day Type Index [](Hourly)']).mean()})
weekday.index = ['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom']

plt.bar(x=weekday.index, height=weekday['medium'], color='b', label='Medium office')
plt.bar(x=weekday.index, height=weekday['small'], color='r', label='Small office')
plt.title('Total cooling rate [W]')
plt.yticks(np.arange(0, 1.1, 0.1))
plt.grid()
plt.legend()
plt.savefig('def_code/immagini/cooling_rate_frequency.png')
plt.show()


# DENSITY (COOLING RATE)
# Libraries & dataset
import seaborn as sns
import matplotlib.pyplot as plt

# set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above)
sns.set(style="darkgrid")
# df = sns.load_dataset('iris')

weekday.index = [1,2,3,4,5,6,7]

# plotting both distibutions on the same figure
fig = sns.kdeplot(y=weekday['medium'], shade=True, color="r", label='Medium office')
fig = sns.kdeplot(y=weekday['small'], shade=True, color="b", label='Small office')
plt.title('Cooling rate density distribution', size=15)
plt.legend()
plt.show()

"""
# ______________________________________Datasets_preprocessing__________________________________________________________
# shifting_period = 1
period = 1
l_train = int(0.5 * len(medium_office))
l_val = int(l_train+2928)
# l_test = int(len(medium_office)-l_val)

#l_train_m = int(0.8 * l_train)# training length

def create_data(df, col_name):
    train_mx = pd.DataFrame(df[:l_train])
    val_mx = pd.DataFrame(df[l_train:l_val])
    # val_mx = pd.DataFrame(df[l_train_m:l_train])
    test_mx = pd.DataFrame(df[l_val:])
    # test_mx = pd.DataFrame(df[l_train:])
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
train_m, val_m, test_m = train_m.to_numpy(), val_m.to_numpy(), test_m.to_numpy()
# train_s, val_s, test_s = create_data(df=small_office, col_name='Mean air Temperature [°C]')
# train_s, val_s, test_s = train_s.to_numpy(), val_s.to_numpy(), test_s.to_numpy()

#_____________________________________Split_the_x_and_y_datasets________________________
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

# Split medium office
train_mX, train_mY = split_sequences(sequences=train_m, n_steps=n_steps)
val_mX, val_mY = split_sequences(sequences=val_m, n_steps=n_steps)
test_mX, test_mY = split_sequences(sequences=test_m, n_steps=n_steps)

# Convert medium office to tensors
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


# Split small office
"""
train_sX, train_sY = split_sequences(sequences=train_s, n_steps=n_steps)
val_sX, val_sY = split_sequences(sequences=val_s, n_steps=n_steps)
test_sX, test_sY = split_sequences(sequences=test_s, n_steps=n_steps)

# Convert small office to tensors
train_sX = torch.from_numpy(train_sX)
train_sY = torch.from_numpy(train_sY)
val_sX = torch.from_numpy(val_sX)
val_sY = torch.from_numpy(val_sY)
test_sX = torch.from_numpy(test_sX)
test_sY = torch.from_numpy(test_sY)

print(type(train_sX), train_sX.shape)
print(type(train_sY), train_sY.shape)
print(type(val_sX), val_sX.shape)
print(type(val_sY), val_sY.shape)
print(type(test_sX), test_sX.shape)
print(type(test_sY), test_sY.shape)

"""


#======================================== LSTM Structure ========================================#
#HYPER PARAMETERS
lookback = 48
# train_episodes = 25
lr = 0.008 #0.005 #0.009
num_hidden = 5
num_layers = 3

num_hidden1 = 5
num_layers1 = 3

# batch_size = 100

train_batch_size = 500
train_data_m = TensorDataset(train_mX, train_mY)
train_dl_m = DataLoader(train_data_m, batch_size=train_batch_size, shuffle=True, drop_last=True)

val_batch_size = 200
val_data_m = TensorDataset(val_mX, val_mY)
val_dl_m = DataLoader(val_data_m, batch_size=val_batch_size, shuffle=True, drop_last=True)

# Structure
class MV_LSTM(torch.nn.Module):
    def __init__(self, n_features, seq_length, drop_prob=0.2):
        super(MV_LSTM, self).__init__()
        self.seq_len = seq_length
        self.n_hidden = num_hidden # number of hidden states
        self.n_layers = num_layers # number of LSTM layers (stacked)

        self.n_hidden1 = num_hidden1
        self.n_layers1 = num_layers1

        self.l_lstm = torch.nn.LSTM(input_size = n_features,
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers,
                                 batch_first = True)

        self.l_lstm1 = torch.nn.LSTM(input_size=self.n_hidden,
                                    hidden_size=self.n_hidden1,
                                    num_layers=self.n_layers1,
                                    batch_first=True)
        # self.dropout = torch.nn.Dropout(drop_prob)
        # according to pytorch docs LSTM output isn(batch_size,seq_len, num_directions * hidden_size) when considering batch_first = True
        self.l_linear = torch.nn.Sequential(
            nn.Linear(5, 3),
            nn.ReLU(),
            nn.Linear(3, 1)
            # nn.ReLU()
            #nn.Linear(7, 3),
            #nn.ReLU(),
            #nn.Linear(3, 1)
        )

    def forward(self, x, h):
        batch_size, seq_len, _ = x.size()
        # hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        # cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        lstm_out, h = self.l_lstm(x, h)
        lstm_out, h = self.l_lstm1(lstm_out, h)

        # h_out_numpy = h[0].detach().numpy() # se n layer = 1 all'ora h_out_numpy è uguale a out_numpy2
        # out_numpy = lstm_out.detach().numpy()
        out = lstm_out[:, -1, :]
        # out_numpy2 = out.detach().numpy()#many to one, I take only the last output vector, for each Batch
        out_linear_transf = self.l_linear(out)
        return out_linear_transf, h

    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        hidden = (hidden_state, cell_state)  # HIDDEN is defined as a TUPLE
        return hidden

"""
    def forward(self, input):
        h_t1 = Variable(torch.zeros(self.n_layers, input.size()[1], self.n_hidden))
        c_t1 = Variable(torch.zeros(self.n_layers, input.size()[1], self.n_hidden))
        h_t2 = Variable(torch.zeros(self.n_layers1, input.size()[1], self.n_hidden1))
        c_t2 = Variable(torch.zeros(self.n_layers1, input.size()[1], self.n_hidden1))
        outputs = []

        for i, input_t in enumerate(input.chunk(input.size(1))):
            _, (h_t1, c_t1) = self.lstm(input_t, (h_t1, c_t1))
            _, (h_t2, _) = self.lstm_1(h_t1, (h_t2, c_t2))
            output = self.linear(h_t2[-1])
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)

        return outputs

"""

# Create NN
#generalize the number of features and the number of timesteps by linking them to the preprocessing
n_features = train_mX.shape[2]
n_timesteps = lookback

#initialize the network,criterion and optimizer
lstm = MV_LSTM(n_features, n_timesteps)
criterion_m = torch.nn.MSELoss() # reduction='sum' created huge loss value
optimizer_m = torch.optim.Adam(lstm.parameters(), lr=lr)


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
            label = label.unsqueeze(1) # utilizzo .unsqueeze per non avere problemi di dimensioni
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


epochs_m = 90
train_loss_m, val_loss_m = train_model(lstm, epochs=epochs_m, train_dl=train_dl_m, val_dl=val_dl_m, optimizer=optimizer_m, criterion=criterion_m)


# Plot to verify validation and train loss, in order to avoid underfitting and overfitting
plt.plot(train_loss_m,'--',color='r', linewidth = 1, label = 'Train Loss')
plt.plot(val_loss_m,color='b', linewidth = 1, label = 'Validation Loss')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
plt.xticks(np.arange(0, int(epochs_m), 20))
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.title("Training VS Validation loss", size=15)
plt.legend()
# plt.savefig('def_code/immagini/LSTM/hidden_freezed/7_7_3/LSTM_Train_VS_Val_LOSS({}_epochs).png'.format(epochs_m))
plt.show()

# _____________________________________________________SAVE THE MODEL ____________________________________________________


torch.save(lstm.state_dict(), 'def_code/lstm_(freezed=7,7,3)_medium_office.pth')
model = MV_LSTM(n_features, n_timesteps)
model.load_state_dict(torch.load('def_code/lstm_(freezed=7,7,3)_medium_office.pth'))
model.eval()



# __________________________________________________1h PREDICTION TESTING__________________________________________
test_batch_size = 200
test_data_m = TensorDataset(test_mX, test_mY)
test_dl_m = DataLoader(test_data_m, shuffle=False, batch_size=test_batch_size, drop_last=True)
test_losses = []
# h = lstm.init_hidden(val_batch_size)

def test_model(model, test_dl, maxT, minT, batch_size):
    h = model.init_hidden(batch_size)
    #model.eval()
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

y_pred_m, y_lab_m = test_model(lstm, test_dl_m, maxT_m, minT_m, test_batch_size)

flatten = lambda l: [item for sublist in l for item in sublist]
y_pred_m = flatten(y_pred_m)
y_lab_m = flatten(y_lab_m)
y_pred_m = np.array(y_pred_m, dtype=float)
y_lab_m = np.array(y_lab_m, dtype=float)


error_m = []
error_m = y_pred_m - y_lab_m

plt.hist(error_m, 100, linewidth=1.5, edgecolor='black', color='orange')
plt.xticks(np.arange(-0.4, 0.4, 0.1))
plt.xlim(-0.4, 0.4)
plt.title('LSTM model prediction error')
# plt.xlabel('Error')
plt.grid(True)
#plt.savefig('def_code/immagini/LSTM/hidden_freezed/7_7_3/LSTM_model_error({}_epochs).png'.format(epochs_m))
plt.show()


plt.plot(y_pred_m, color='orange', label="Predicted")
plt.plot(y_lab_m, color="b", linestyle="dashed", linewidth=1, label="Real")
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlim(left=0, right=800)
plt.ylabel('Mean Air Temperature [°C]')
plt.xlabel('Time [h]')
plt.title("Real VS predicted temperature", size=15)
plt.legend()
#plt.savefig('def_code/immagini/LSTM/hidden_freezed/7_7_3/LSTM_real_VS_predicted_temperature({}_epochs).png'.format(epochs_m))
plt.show()


# METRICS
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

MAPE = mean_absolute_percentage_error(y_lab_m, y_pred_m)
RMSE = mean_squared_error(y_lab_m,y_pred_m)**0.5
R2 = r2_score(y_lab_m,y_pred_m)

print('MAPE:%0.5f%%'%MAPE)
print('RMSE:', RMSE.item())
print('R2:', R2.item())


plt.scatter(y_lab_m, y_pred_m,  color='k', edgecolor= 'white', linewidth=1,alpha=0.1)
plt.text(25.5, 29.2, 'MAPE: {:.3f}'.format(MAPE), fontsize=15, bbox=dict(facecolor='red', alpha=0.5))
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlabel('Real Temperature [°C]')
plt.ylabel('Predicted Temperature [°C]')
plt.title("Prediction distribution", size=15)
#plt.savefig('def_code/immagini/LSTM/hidden_freezed/7_7_3/LSTM_prediction_distribution({}_epochs).png'.format(epochs_m))
plt.show()


# _____________________________________________________TUNING_PHASE_____________________________________________________

def freeze_params(model):
    for param_c in model.l_lstm.parameters():
            param_c.requires_grad = False
    for param_fc in model.l_linear.parameters():
            param_fc.requires_grad = False
    return model

# for param_c in mv_net.l_lstm.parameters():
#     print(param_c)
lstm_test = freeze_params(model)
#lstm_test.l_linear = nn.Sequential(*list(lstm_test.l_linear.children())[:-2])
# lstm_tun = freeze_params(lstm_prova)
print(lstm_test)
for i in lstm_test.l_lstm1.parameters():
    print(i)
for x in lstm_test.l_linear.parameters():
    print(x)

# ______________________________ADD MODULES_____________________________________________________________________________
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

# __________________________________INCLUDE_NEW_DATASET_________________________________________________________________
# from new_dataset import train_mX_new, train_mY_new, val_mX_new, val_mY_new, test_mX_new, test_mY_new
from one_month_small import train_small_1mX, train_small_1mY, val_small_1mX, val_small_1mY, test_small_1mX, test_small_1mY, maxT_small_1m, minT_small_1m

train_batch_size = 90
train_data_small_1m = TensorDataset(train_small_1mX, train_small_1mY)
train_dl_small_1m = DataLoader(train_data_small_1m, batch_size=train_batch_size, shuffle=True, drop_last=True)

val_batch_size = 18
val_data_small_1m = TensorDataset(val_small_1mX, val_small_1mY)
val_dl_small_1m = DataLoader(val_data_small_1m, batch_size=val_batch_size, shuffle=True, drop_last=True)


# generalize the number of features and the number of timesteps by linking them to the preprocessing
n_features = train_small_1mX.shape[2]
n_timesteps = lookback

# initialize the network,criterion and optimizer
criterion_ft = torch.nn.MSELoss()
optimizer_ft = torch.optim.SGD(lstm.parameters(), lr=0.001)
#optimizer_ft = torch.optim.SGD(filter(lambda p: p.requires_grad, lstm_test.parameters()), lr=0.001)
# Decay LR (learning rate) by a factor of 0.1 every 7 epochs
lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

# TRAINING TUNING MODEL
epochs_s = 120
train_loss_st_1m, val_loss_st_1m = train_model(lstm, epochs_s, train_dl_small_1m, val_dl_small_1m, optimizer_ft, criterion_ft, mode='tuning')


# Plot to verify validation and train loss, in order to avoid underfitting and overfitting
plt.plot(train_loss_st_1m, '--', color='r', linewidth=1, label='Train Loss')
plt.plot(val_loss_st_1m, color='b', linewidth=1, label='Validation Loss')
plt.ylabel('Loss (MSE)')
#plt.ylim(0, 0.005)
plt.xlabel('Epoch')
plt.xticks(np.arange(0, int(epochs_s), 10))
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.title("Training VS Validation loss", size=15)
plt.legend()
#plt.savefig('def_code/immagini/LSTM/hidden_freezed/7_7_3/tuning/center_training/LSTM_tuning_Train_VS_Val_LOSS({}_epochs).png'.format(epochs_s))
plt.show()

# ______________________________________TESTING______________________________
test_batch_size = 20
test_data_s = TensorDataset(test_small_1mX, test_small_1mY)
test_dl_s = DataLoader(test_data_s, shuffle=False, batch_size=test_batch_size, drop_last=True)
test_losses_s = []
# h = lstm.init_hidden(val_batch_size)

y_pred_st_1m, y_lab_st_1m = test_model(lstm, test_dl_s, maxT_small_1m, minT_small_1m, test_batch_size)

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
#plt.savefig('def_code/immagini/LSTM/hidden_freezed/7_7_3/tuning/center_training/LSTM_tuning_model_error({}_epochs).png'.format(epochs_s))
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
#plt.savefig('def_code/immagini/LSTM/hidden_freezed/7_7_3/tuning/center_training/LSTM_tuning_real_VS_predicted_temperature({}_epochs).png'.format(epochs_s))
plt.show()


MAPE = mean_absolute_percentage_error(y_lab_st_1m, y_pred_st_1m)
RMSE = mean_squared_error(y_lab_st_1m, y_pred_st_1m)**0.5
R2 = r2_score(y_lab_st_1m, y_pred_st_1m)

print('MAPE:%0.5f%%'%MAPE)
print('RMSE:', RMSE.item())
print('R2:', R2.item())


plt.scatter(y_lab_st_1m, y_pred_st_1m,  color='k', edgecolor= 'white', linewidth=1) # ,alpha=0.1
plt.text(25.2, 27.7, 'MAPE: {:.3f}'.format(MAPE), fontsize=15, bbox=dict(facecolor='red', alpha=0.5))
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlabel('Real Temperature [°C]')
plt.ylabel('Predicted Temperature [°C]')
plt.title("Tuning prediction distribution", size=15)
#plt.savefig('def_code/immagini/LSTM/hidden_freezed/7_7_3/tuning/center_training/LSTM_tuning_prediction_distribution({}_epochs).png'.format(epochs_s))
plt.show()

