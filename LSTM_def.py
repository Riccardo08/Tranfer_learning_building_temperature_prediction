import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime as dt
import os
import numpy as np
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

# Normalization
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


# Split the x and y datasets
#======================================================
n_steps = 48
train_mX, train_mY = split_sequences(train_m, n_steps=n_steps)
val_mX, val_mY = split_sequences(val_m, n_steps=n_steps)
test_mX, test_mY = split_sequences(test_m, n_steps=n_steps)

# Convert to tensors
train_mX=torch.from_numpy(train_mX)
train_mY=torch.from_numpy(train_mY)
val_mX=torch.from_numpy(val_mX)
val_mY=torch.from_numpy(val_mY)
test_mX=torch.from_numpy(test_mX)
test_mY=torch.from_numpy(test_mY)

print(type(train_mX), train_mX.shape)
print(type(train_mY), train_mY.shape)
print(type(val_mX), val_mX.shape)
print(type(val_mY), val_mY.shape)
print(type(test_mX), test_mX.shape)
print(type(test_mY), test_mY.shape)


#======================================== LSTM Structure ========================================#
#HYPER PARAMETERS
lookback = 48
train_episodes = 10
lr = 0.008 #0.005 #0.009
num_layers = 5
num_hidden = 8
batch_size = 100


train_batch_size = 500
train_data = TensorDataset(train_mX, train_mY)
train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

val_batch_size = 300
val_data = TensorDataset(val_mX, val_mY)
val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=True, drop_last=True)

test_data = TensorDataset(test_mX, test_mY)
test_dl = DataLoader(test_data) # batch_size -> terza dimensione

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
        # according to pytorch docs LSTM output is
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_linear = torch.nn.Linear(self.n_hidden, 1)

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
n_features = train_mX.shape[2]
n_timesteps = lookback

#initialize the network,criterion and optimizer
mv_net = MV_LSTM(n_features, n_timesteps)
criterion = torch.nn.MSELoss() # reduction='sum' created huge loss value
optimizer = torch.optim.Adam(mv_net.parameters(), lr=lr)

#initialize the training loss and the validation loss
LOSS = []
VAL_LOSS = []

#START THE TRAINING PROCESS
mv_net.train()

for t in range(train_episodes):

    h = mv_net.init_hidden(batch_size)  #hidden state is initialized at each epoch
    loss = []
    for x, label in train_dl:
        h = mv_net.init_hidden(batch_size) #since the batch is big enough, a stateless mode is used (also considering the possibility to shuffle the training examples, which increase the generalization ability of the network)
        h = tuple([each.data for each in h])
        output, h = mv_net(x.float(), h)
        label = label.unsqueeze(1) #utilizzo .unsqueeze per non avere problemi di dimensioni
        loss_c = criterion(output, label.float())
        optimizer.zero_grad()
        loss_c.backward()
        optimizer.step()
        loss.append(loss_c.item())
    LOSS.append(np.sum(loss) /batch_size)
    # print("Epoch: %d, training loss: %1.5f" % (train_episodes, LOSS[-1]))


    # VALIDATION LOOP
    val_loss =[]
    h = mv_net.init_hidden(batch_size)
    for inputs, labels in val_dl:
        h = tuple([each.data for each in h])
        val_output, h = mv_net(inputs.float(), h)
        val_labels = labels.unsqueeze(1)
        val_loss_c = criterion(val_output, val_labels.float())
        val_loss.append(val_loss_c.item())
    # VAL_LOSS.append(val_loss.item())
    VAL_LOSS.append(np.sum(val_loss) /batch_size)
    print('Epoch : ', t, 'Training Loss : ', LOSS[-1], 'Validation Loss :', VAL_LOSS[-1])
    #print("Epoch: %d, training loss: %1.5f" % (train_episodes, VAL_LOSS[-1]))



#Plot to verify validation and train loss, in order to avoid underfitting and overfitting
plt.plot(LOSS,'--',color='r', linewidth = 1, label = 'Train Loss')
plt.plot(VAL_LOSS,color='b', linewidth = 1, label = 'Validation Loss')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
plt.xticks(np.arange(0, 10, 1))
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.title("Training VS Validation loss", size=15)
plt.legend()
# plt.savefig('immagini_LSTM/I_LSTM_Train_VS_Val_LOSS(10_epochs).png')
plt.show()




#=========================================================================================#
#1h PREDICTION TESTING
test_data = TensorDataset(test_mX, test_mY)
test_dl = DataLoader(test_data, shuffle=False, batch_size=batch_size, drop_last=True)
test_losses = []
h = mv_net.init_hidden(batch_size)


mv_net.eval()
ypred=[]
ylab=[]
for inputs, labels in test_dl:
    h = tuple([each.data for each in h])
    test_output, h = mv_net(inputs.float(), h)
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
    ypred.append(test_output)
    ylab.append(labels)

flatten = lambda l: [item for sublist in l for item in sublist]
ypred = flatten(ypred)
ylab = flatten(ylab)
ypred = np.array(ypred, dtype=float)
ylab = np.array(ylab, dtype = float)


error = []
error = ypred - ylab

plt.hist(error, 50, linewidth=1.5, edgecolor='black', color='orange')
plt.xticks(np.arange(-0.4, 0.4, 0.1))
plt.xlim(-0.4, 0.4)
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
plt.xlim(left=0,right=800)
plt.ylabel('Mean Air Temperature [°C]')
plt.xlabel('Time [h]')
plt.title("Real VS predicted temperature", size=15)
plt.legend()
# plt.savefig('immagini_LSTM/I_LSTM_real_VS_predicted_temperature(10_epochs).png')
plt.show()


#METRICS
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

MAPE = mean_absolute_percentage_error(ylab, ypred)
RMSE=mean_squared_error(ylab,ypred)**0.5
R2 = r2_score(ylab,ypred)

print('MAPE:%0.5f%%'%MAPE)
print('RMSE:', RMSE.item())
print('R2:', R2.item())


plt.scatter(ylab,ypred,  color='k', edgecolor= 'white', linewidth=1,alpha=0.1)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlabel('Real Temperature [°C]')
plt.ylabel('Predicted Temperature [°C]')
plt.title("Prediction distribution", size=15)
# plt.savefig('immagini_LSTM/I_LSTM_prediction_distribution(10_epochs).png')
plt.show()


"""
#========================= SENSITIVITY ANALYSIS ==================================#

#HYPER PARAMETERS
lookback = lookback
train_episodes = 10
batch_size=100
lr = 0.008 # se analizzo un altro parametro, il vettore conterrà elementi di un altro parametro, per esempio lr
num_hidden_vec=np.array([1,8,10,20,30,50,100])
num_layers = 5
n_hidden = 8



#Creo la rete neurale
class MV_LSTM(torch.nn.Module):
    def __init__(self,n_features,seq_length,num_hidden,num_layers):
        super(MV_LSTM, self).__init__()
        self.seq_len = seq_length
        self.n_hidden = num_hidden # number of hidden states
        self.n_layers = num_layers # number of LSTM layers (stacked)
        self.l_lstm = torch.nn.LSTM(input_size = n_features,
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers,
                                 batch_first = True)
        # according to pytorch docs LSTM output is
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_linear = torch.nn.Linear(self.n_hidden, 1)


    def forward(self, x,h):
        batch_size, seq_len, _ = x.size()
        lstm_out, h = self.l_lstm(x,h)
        #out_numpy = lstm_out.detach().numpy()
        out = lstm_out[:,-1,:]  #many to one, I take only the last output vector, for each Batch
        out_linear_transf = self.l_linear(out)
        return out_linear_transf, h

    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        cell_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        hidden = (hidden_state, cell_state) #HIDDEN VIENE DEFINITO COME UNA TUPLE
        return hidden


repeat = 10 #quante ripeto il processo per ogni singolo componente nel vettore del parametro considerato

rmse_scores = np.zeros((repeat,len(num_hidden_vec)))
R2_scores = np.zeros((repeat,len(num_hidden_vec)))
MAPE_scores = np.zeros((repeat,len(num_hidden_vec)))

for index in range(len(num_hidden_vec)):
    print(index)

    for r in range(repeat):
        n_hidden = int(num_hidden_vec[index])
        # UTILIZZO IL DATA LOADER PER CARICARE I GLI INPUT E I LABEL DEL TRAIN SET
        train_data = TensorDataset(train_mX, train_mY)
        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
        test_data = TensorDataset(test_mX, test_mY)
        test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, drop_last=True)
        # create NN
        # generalizzo il numero di features e il numero di timesteps collegandolo al preprocessing
        n_features = train_mX.shape[2]
        n_timesteps = lookback
        # inizializzo rete, criterio e ottimizzatore
        mv_net = MV_LSTM(n_features, n_timesteps, n_hidden, num_layers)
        # mv_net = mv_net.to(device)
        criterion = torch.nn.MSELoss()  # reduction='sum' created huge loss value
        LOSS = []
        ypred_train = []
        ylab_train = []
        rmse_train = []
        rmse_test = []

        for t in range(train_episodes):
            mv_net.train()
            optimizer = torch.optim.Adam(mv_net.parameters(), lr)
            # h = mv_net.init_hidden(batch_size)  # VADO AD INIZIALIZZARE HIDDEN STATE E CELL STATE SOLO AD OGNI EPOCA
            for x, label in train_loader:
                h = mv_net.init_hidden(batch_size)
                h = tuple([each.data for each in h]) # VADO AD INIZIALIZZARE L'HIDDEN STATE E IL CELL STATE PER OGNI BATCH COME L'HIDDEN ED IL CELL DELLA BATCH PRECEDENTE, EVITANDO COSI' DI PERDERE DELLE INFORMAZIONI RILEVANTI
                output, h = mv_net(x.float(), h)
                label = label.unsqueeze(1) #utilizzo .unsqueeze per non avere problemi di dimensioni
                loss = criterion(output, label.float().reshape(-1,1))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                #TRANSFORM EACH OUTPUT FROM A TENSOR TO A NUMPY ARRAY
                output = output.detach().numpy()
                output = np.reshape(output, (-1, 1))
                output = minT + output*(maxT-minT)

                #TRANSFORM EACH LABEL FROM A TENSOR TO A NUMPY ARRAY
                label = label.detach().numpy()
                label = np.reshape(label, (-1, 1))
                label = minT + label*(maxT-minT)
                ypred_train.append(output)
                ylab_train.append(label)
            LOSS.append(loss.item())



        h = mv_net.init_hidden(batch_size)
        mv_net.eval()
        ypred=[]
        ylab=[]
        for inputs, labels in test_loader:
            h = tuple([each.data for each in h])
            test_output, h = mv_net(inputs.float(), h)
            labels = labels.unsqueeze(1)
            test_output = test_output.detach().numpy()
            test_output = np.reshape(test_output, (-1, 1))
            test_output = minT + test_output*(maxT-minT)
            labels = labels.detach().numpy()
            labels = np.reshape(labels, (-1, 1))
            labels = minT + labels*(maxT-minT)
            ypred.append(test_output)
            ylab.append(labels)
        flatten = lambda l: [item for sublist in l for item in sublist]
        ypred = flatten(ypred)
        ylab = flatten(ylab)
        yl = np.array(ylab)
        yp = np.array(ypred)


        rmse_scores[r,index] = mean_squared_error(yl,yp)**0.5
        R2_scores[r,index] = r2_score(yl,yp)
        MAPE_scores[r,index] = mean_absolute_percentage_error(yl,yp)



#=============================== SUMMARIZE RESULTS =============================
# print(rmse_scores.describe())
# print(R2_scores.describe())
# print(MAPE_scores.describe())
RMSE = pd.DataFrame(rmse_scores)
R2 = pd.DataFrame(R2_scores)
MAPE = pd.DataFrame(MAPE_scores)

#===============================================================================
#salvo i risultati in un excel per poterli poi analizzare
RMSE.to_excel('RMSE.xlsx', encoding='utf-8', index=False)
MAPE.to_excel('MAPE.xlsx', encoding='utf-8', index=False)
R2.to_excel('R2.xlsx', encoding='utf-8', index=False)


#creo i BOXPLOT per l'hyperparameter considerato, considerando R2, MAPE e RMSE
vals, xs = [],[]
for i, col in enumerate(MAPE.columns):
     vals.append(MAPE[col].values)
     xs.append(np.random.normal(i+1, 0.05, MAPE[col].values.shape[0]))  # adds jitter to the data points - can be adjusted

plt.boxplot(vals, labels = num_hidden_vec, showmeans=True, meanline=True)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.05)
plt.ylabel('MAPE [%]')
plt.xlabel('Neurons per layer')
plt.title("MAPE", size=15)
palette = ['r', 'g', 'b', 'y','orange','deepskyblue','magenta']
for x, val, c in zip(xs, vals, palette):
     plt.scatter(x, val, alpha=0.4, color=c)
# plt.savefig('immagini_LSTM/MAPE_distribution(sensitivity).png')
plt.show()


vals, xs = [],[]
for i, col in enumerate(RMSE.columns):
     vals.append(RMSE[col].values)
     xs.append(np.random.normal(i+1, 0.05, RMSE[col].values.shape[0]))  # adds jitter to the data points - can be adjusted

plt.boxplot(vals, labels = num_hidden_vec, showmeans=True, meanline=True)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.05)
plt.ylabel('RMSE [-]')
plt.xlabel('Neurons per layer')
plt.title("RMSE", size=15)
for x, val, c in zip(xs, vals, palette):
     plt.scatter(x, val, alpha=0.4, color=c)
# plt.savefig('immagini_LSTM/RMSE_distribution(sensitivity).png')
plt.show()

vals, xs = [],[]
for i, col in enumerate(R2.columns):
     vals.append(R2[col].values)
     xs.append(np.random.normal(i+1, 0.05, R2[col].values.shape[0]))  # adds jitter to the data points - can be adjusted

plt.boxplot(vals, labels = num_hidden_vec, showmeans=True, meanline=True)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.05)
plt.ylabel('R2 [-]')
plt.xlabel('Neurons per layer')
plt.title("R2", size=15)
for x, val, c in zip(xs, vals, palette):
     plt.scatter(x, val, alpha=0.4, color=c)
# plt.savefig('immagini_LSTM/R2_distribution(sensitivity).png')
plt.show()
"""


#======================================== LSTM Structure with 10 neurons per layer and 10 epochs ========================================#
#HYPER PARAMETERS
lookback = 48
train_episodes = 10
lr = 0.008 #0.005 #0.009
num_layers = 5
num_hidden = 10
batch_size = 100


train_batch_size = 500
train_data = TensorDataset(train_mX, train_mY)
train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

val_batch_size = 300
val_data = TensorDataset(val_mX, val_mY)
val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=True, drop_last=True)

test_data = TensorDataset(test_mX, test_mY)
test_dl = DataLoader(test_data) # batch_size -> terza dimensione

# Structure
class MV_LSTM(torch.nn.Module):
    def __init__(self, n_features, seq_length, drop_prob=0.2):
        super(MV_LSTM, self).__init__()
        self.seq_len = seq_length
        self.n_hidden = num_hidden# number of hidden states
        self.n_layers = num_layers# number of LSTM layers (stacked)
        self.l_lstm = torch.nn.LSTM(input_size = n_features,
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers,
                                 batch_first = True)
        # self.dropout = torch.nn.Dropout(drop_prob)
        # according to pytorch docs LSTM output is
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_linear = torch.nn.Linear(self.n_hidden, 1)

    def forward(self, x, h):
        batch_size, seq_len, _ = x.size()
        # hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        # cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        lstm_out, h = self.l_lstm(x,h)
        # h_out_numpy = h[0].detach().numpy() # se n layer = 1 all'ora h_out_numpy è ugugla a out_numpy2
        # out_numpy = lstm_out.detach().numpy()
        out = lstm_out[:,-1,:]
        # out_numpy2 = out.detach().numpy()#many to one, I take only the last output vector, for each Batch
        out_linear_transf = self.l_linear(out)
        return out_linear_transf, h

    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        cell_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        hidden = (hidden_state, cell_state) #HIDDEN is defined as a TUPLE
        return hidden


# create NN
#generalize the number of features and the number of timesteps by linking them to the preprocessing
n_features = train_mX.shape[2]
n_timesteps = lookback

#initialize the network,criterion and optimizer
mv_net = MV_LSTM(n_features, n_timesteps)
criterion = torch.nn.MSELoss() # reduction='sum' created huge loss value
optimizer = torch.optim.Adam(mv_net.parameters(), lr=lr)

#initialize the training loss and the validation loss
LOSS = []
VAL_LOSS = []

#START THE TRAINING PROCESS
mv_net.train()

for t in range(train_episodes):

    h = mv_net.init_hidden(batch_size)  #hidden state is initialized at each epoch
    loss = []
    for x, label in train_dl:
        h = mv_net.init_hidden(batch_size) #since the batch is big enough, a stateless mode is used (also considering the possibility to shuffle the training examples, which increase the generalization ability of the network)
        h = tuple([each.data for each in h])
        output, h = mv_net(x.float(), h)
        label = label.unsqueeze(1) #utilizzo .unsqueeze per non avere problemi di dimensioni
        loss_c = criterion(output, label.float())
        optimizer.zero_grad()
        loss_c.backward()
        optimizer.step()
        loss.append(loss_c.item())
    LOSS.append(np.sum(loss) /batch_size)
    # print("Epoch: %d, training loss: %1.5f" % (train_episodes, LOSS[-1]))


    # VALIDATION LOOP
    val_loss =[]
    h = mv_net.init_hidden(batch_size)
    for inputs, labels in val_dl:
        h = tuple([each.data for each in h])
        val_output, h = mv_net(inputs.float(), h)
        val_labels = labels.unsqueeze(1)
        val_loss_c = criterion(val_output, val_labels.float())
    # VAL_LOSS.append(val_loss.item())
        val_loss.append(val_loss_c.item())
    VAL_LOSS.append(np.sum(val_loss) /batch_size)
    print('Epoch : ', t, 'Training Loss : ', LOSS[-1], 'Validation Loss :', VAL_LOSS[-1])
    #print("Epoch: %d, training loss: %1.5f" % (train_episodes, VAL_LOSS[-1]))



#Plot to verify validation and train loss, in order to avoid underfitting and overfitting
plt.plot(LOSS,'--',color='r', linewidth = 1, label = 'Train Loss')
plt.plot(VAL_LOSS,color='b', linewidth = 1, label = 'Validation Loss')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
plt.xticks(np.arange(0, 10, 1))
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.title("Training VS Validation loss", size=15)
plt.legend()
plt.savefig('immagini_LSTM/final_LSTM_Train_VS_Val_LOSS(10_neurons).png')
plt.show()




#=========================================================================================#
#1h PREDICTION TESTING
test_data = TensorDataset(test_mX, test_mY)
test_dl = DataLoader(test_data, shuffle=False, batch_size=batch_size, drop_last=True)
test_losses = []
h = mv_net.init_hidden(batch_size)


mv_net.eval()
ypred=[]
ylab=[]
for inputs, labels in test_dl:
    h = tuple([each.data for each in h])
    test_output, h = mv_net(inputs.float(), h)
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
    ypred.append(test_output)
    ylab.append(labels)

flatten = lambda l: [item for sublist in l for item in sublist]
ypred = flatten(ypred)
ylab = flatten(ylab)
ypred = np.array(ypred, dtype=float)
ylab = np.array(ylab, dtype=float)



final_error = []
final_error = ypred - ylab

plt.hist(final_error, 50, linewidth=1.5, edgecolor='black', color='lime')
plt.xticks(np.arange(-0.4, 0.4, 0.1))
plt.xlim(-0.4, 0.4)
plt.title('Final model prediction error')
# plt.xlabel('Error')
plt.grid(True)
# plt.savefig('immagini_LSTM/final_model_error.png')
plt.show()


plt.plot(ypred, color='orange', label="Predicted")
plt.plot(ylab, color="b", linestyle="dashed", linewidth=1, label="Real")
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlim(left=0,right=360)
plt.ylabel('Mean Air Temperature [°C]')
plt.xlabel('Time [h]')
plt.title("Real VS predicted temperature", size=15)
plt.legend()
plt.savefig('immagini_LSTM/final_LSTM_real_VS_predicted_temperature(10_neurons).png')
plt.show()


#METRICS
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

MAPE = mean_absolute_percentage_error(ylab, ypred)
RMSE=mean_squared_error(ylab,ypred)**0.5
R2 = r2_score(ylab,ypred)

print('MAPE:%0.5f%%'%MAPE)
print('RMSE:%0.5f'%RMSE.item())
print('R2:%0.5f'%R2.item())

plt.scatter(ylab,ypred,  color='k', edgecolor= 'white', linewidth=1,alpha=0.1)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlabel('Real Temperature [°C]')
plt.ylabel('Predicted Temperature [°C]')
plt.title("Prediction distribution", size=15)
plt.savefig('immagini_LSTM/final_LSTM_prediction_distribution(10_neurons).png')
plt.show()


#_____________________________________________________TUNING_PHASE_______________________________________________
def freeze_params(model):
    for param_c in model.l_lstm.parameters():
            param_c.requires_grad = False
    for param_fc in model.l_linear.parameters():
            param_fc.requires_grad = True
    return model

# for param_c in mv_net.l_lstm.parameters():
#     print(param_c)

lstm_test = freeze_params(mv_net)
print(lstm_test)
for i in lstm_test.l_lstm.parameters():
    print(i)
for x in lstm_test.l_linear.parameters():
    print(x)

#____________________ADD MODULES_____________________________________________________________________________

lstm_test.l_lstm.add_module('lstm_h', nn.LSTM(input_size=8, hidden_size=num_hidden, num_layers=num_layers, batch_first=True))
#lstm_test.l_lstm.lstm_h = nn.LSTM(input_size=8, hidden_size=num_hidden, num_layers=num_layers, batch_first=True)

print(lstm_test)

num_ftrs = lstm_test.l_linear.in_features
lstm_test.l_linear = nn.Sequential(
    nn.Linear(num_ftrs, 50),
    nn.ReLU(),
    nn.Linear(50, 30),
    nn.ReLU(),
    nn.Linear(30, 1)
)

# How to delete some layers from the model:
# cnn_test.fc = nn.Sequential(*[cnn_test.fc[i] for i in range(4, len(cnn_test.fc))])

criterion_ft = torch.nn.MSELoss()
optimizer_ft = torch.optim.SGD(lstm_test.parameters(), lr=lr)

# Decay LR (learning rate) by a factor of 0.1 every 7 epochs
lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

LOSS = []
VAL_LOSS = []

#START THE TRAINING PROCESS
lstm_test.train()

for t in range(train_episodes):

    h = mv_net.init_hidden(batch_size)  #hidden state is initialized at each epoch
    loss = []
    for x, label in train_dl:
        h = mv_net.init_hidden(batch_size) #since the batch is big enough, a stateless mode is used (also considering the possibility to shuffle the training examples, which increase the generalization ability of the network)
        h = tuple([each.data for each in h])
        output, h = lstm_test(x.float(), h)
        label = label.unsqueeze(1) #utilizzo .unsqueeze per non avere problemi di dimensioni
        loss_c = criterion(output, label.float())
        optimizer.zero_grad()
        loss_c.backward()
        optimizer.step()
        loss.append(loss_c.item())
    LOSS.append(np.sum(loss) /batch_size)
    # print("Epoch: %d, training loss: %1.5f" % (train_episodes, LOSS[-1]))
    lr_scheduler.step()

    # VALIDATION LOOP
    val_loss =[]
    h = mv_net.init_hidden(batch_size)
    for inputs, labels in val_dl:
        h = tuple([each.data for each in h])
        val_output, h = lstm_test(inputs.float(), h)
        val_labels = labels.unsqueeze(1)
        val_loss_c = criterion(val_output, val_labels.float())
        val_loss.append(val_loss_c.item())
    # VAL_LOSS.append(val_loss.item())
    VAL_LOSS.append(np.sum(val_loss) /batch_size)
    print('Epoch : ', t, 'Training Loss : ', LOSS[-1], 'Validation Loss :', VAL_LOSS[-1])
    #print("Epoch: %d, training loss: %1.5f" % (train_episodes, VAL_LOSS[-1]))



#Plot to verify validation and train loss, in order to avoid underfitting and overfitting
plt.plot(LOSS,'--',color='r', linewidth = 1, label = 'Train Loss')
plt.plot(VAL_LOSS,color='b', linewidth = 1, label = 'Validation Loss')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
plt.xticks(np.arange(0, train_episodes, 1))
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.title("Training VS Validation loss", size=15)
plt.legend()
# plt.savefig('immagini_LSTM/final_LSTM_Train_VS_Val_LOSS(10_neurons).png')
plt.show()

#______________________________________TESTING______________________________
test_data = TensorDataset(test_mX, test_mY)
test_dl = DataLoader(test_data, shuffle=False, batch_size=batch_size, drop_last=True)
test_losses = []
h = lstm_test.init_hidden(batch_size)


lstm_test.eval()
ypred=[]
ylab=[]
for inputs, labels in test_dl:
    h = tuple([each.data for each in h])
    test_output, h = lstm_test(inputs.float(), h)
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
    ypred.append(test_output)
    ylab.append(labels)

flatten = lambda l: [item for sublist in l for item in sublist]
ypred = flatten(ypred)
ylab = flatten(ylab)
ypred = np.array(ypred, dtype=float)
ylab = np.array(ylab, dtype = float)


error = []
error = ypred - ylab

plt.hist(error, 50, linewidth=1.5, edgecolor='black', color='orange')
plt.xticks(np.arange(-0.4, 0.4, 0.1))
plt.xlim(-0.4, 0.4)
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
plt.xlim(left=0,right=800)
plt.ylabel('Mean Air Temperature [°C]')
plt.xlabel('Time [h]')
plt.title("Real VS predicted temperature", size=15)
plt.legend()
# plt.savefig('immagini_LSTM/I_LSTM_real_VS_predicted_temperature(10_epochs).png')
plt.show()


#METRICS
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

MAPE = mean_absolute_percentage_error(ylab, ypred)
RMSE=mean_squared_error(ylab,ypred)**0.5
R2 = r2_score(ylab,ypred)

print('MAPE:%0.5f%%'%MAPE)
print('RMSE:', RMSE.item())
print('R2:', R2.item())


plt.scatter(ylab,ypred,  color='k', edgecolor= 'white', linewidth=1,alpha=0.1)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlabel('Real Temperature [°C]')
plt.ylabel('Predicted Temperature [°C]')
plt.title("Prediction distribution", size=15)
# plt.savefig('immagini_LSTM/I_LSTM_prediction_distribution(10_epochs).png')
plt.show()
