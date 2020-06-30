from  __future__ import division
from __future__ import  print_function

import numpy as np
import os
import sys
import torch
import torch.nn as nn
import pandas as pd
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import cross_validation
import torch.nn.functional as  F
import torchvision
from torch.utils.data import  Dataset
from torch.utils.data import  DataLoader
import torch
from pandas.core.series import Series
from sklearn import cross_validation
from torchvision import transforms
from matplotlib import cm






subtrainfeature1 = pd.read_csv(r'D:\DeepLearning\Keras_Codel\data\indian_feature_1.csv')
subtrainLabel1 = pd.read_csv(r'D:\DeepLearning\Keras_Codel\data\indian_label_1_101.csv')
print(subtrainfeature1)

subtrain = pd.merge(subtrainLabel1,subtrainfeature1,on='Id')
labels = subtrain.Class
subtrain.drop(["Class","Id"], axis=1, inplace=True)
subtrain = subtrain.as_matrix()



