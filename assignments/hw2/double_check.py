import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')


file_name_1 = 'reion_history_Thesan1.dat'
file_name_2 = 'sfrd_Thesan1.dat'

data1 = pd.read_csv(file_name_1, delim_whitespace=True, skiprows=1, names=['z', 'x_HI', 'x_HeI', 'x_HeII'])
z = data1['z']
x_HI = data1['x_HI']

#print(z)

data11 = data1[(data1['x_HI'] < 0.501) & (data1['x_HI'] > 0.499)]
print('Checking value of z at x_HI == 0.5:\n', data11)

