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
    file = pd.read_csv('datasets/'+directory+'/'+file_csv, encoding='latin1')
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

def concat_datasets(list, columns, name):
    name = pd.DataFrame()
    for x in list:
        name = name.append(x[columns], ignore_index=True)
    # name = pd.concat([list], ignore_index=True)
    return name

medium_office = pd.DataFrame()
small_office = pd.DataFrame()
restaurant = pd.DataFrame()
retail = pd.DataFrame()

medium_office = concat_datasets(list=[medium_office_2_100, medium_office_2_100_random_60_perc, medium_office_2_dataset_validation, medium_office_2_random_2], columns=columns, name=medium_office)
small_office = concat_datasets(list=[small_office_100, small_office_100_random_potenza_60_perc, small_office_105, small_office_random], columns=columns, name=small_office)
restaurant = concat_datasets(list=[restaurant_100, restaurant_100_potenza_random_60_percento, restaurant_dataset_validation, restaurant_random], columns=columns, name=restaurant)
retail = concat_datasets(list=[retail_100, retail_100_potenza_random_60_percento, retail_105, retail_random], columns=columns, name=restaurant)














