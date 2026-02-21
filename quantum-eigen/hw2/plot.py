import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')


# Reading Data
file_name_1 = 'reion_history_Thesan1.dat'
file_name_2 = 'sfrd_Thesan1.dat'

data1 = pd.read_csv(file_name_1, sep=r"\s+", skiprows=1, names=['z', 'x_HI', 'x_HeI', 'x_HeII'])
z = data1['z']
x_HI = data1['x_HI']

data2 = pd.read_csv(file_name_2, delim_whitespace=True, skiprows=1, names=['z', 'sfrd'])
z2 = data2['z']
sfrd = data2['sfrd']

data11 = data1[(data1['x_HI'] < 0.501) & (data1['x_HI'] > 0.499)]
print('Checking value of z at x_HI == 0.5:\n', data11)

# Plot 1: reion.pdf
plt.figure(figsize=(10, 6))
plt.plot(z, x_HI, label='Neutral Hydrogen Fraction (x_HI)', color='blue')
plt.xlabel('Redshift (z)')
plt.ylabel('x_HI')
plt.xlim(5,16)
plt.ylim(0,1.05)
plt.title('Reionization History from Thesan1 Simulation')
plt.savefig('reion.pdf')

# Plot 2: reion_2.pdf
data12 = data1[(data1['x_HI'] <= 1) & (data1['x_HI'] >= 0.1)]
data13 = data1[(data1['x_HI'] <= 0.1) & (data1['x_HI'] >= 0.0001)]

plt.figure(figsize=(10, 6))
plt.plot(data12.z, data12.x_HI, color='red', label='linear')
plt.semilogy(data13.z, data13.x_HI, color='blue', label='log')
plt.xlabel('Redshift (z)')
plt.ylabel('x_HI')
plt.xlim(5,16)
plt.ylim(0.0001,1.05)
plt.legend()
plt.title('Reionization History from Thesan1 Simulation')
plt.savefig('reion_2.pdf')

# Plot 3: sfrd.pdf
plt.figure(figsize=(10, 6))
plt.plot(z2, sfrd, label='Star Formation Rate Density (sfrd)', color='blue')
plt.xlabel('Redshift (z)')
plt.ylabel('sfrd')
plt.ylim(0,0.03)
plt.xlim(5,16)
plt.title('Star Formation Rate Density from Thesan1 Simulation')
plt.savefig('sfrd.pdf')

# Plot 4: sfrd_HI.pdf
updated_HI = interp1d(z, x_HI, kind='linear')(z2)

plt.figure(figsize=(10, 6))
plt.plot(sfrd, updated_HI, label='Interpolated x_HI', color='red')
plt.xlabel('Star Formation Rate Density (sfrd)')
plt.ylabel('Neutral Hydrogen Fraction (x_HI)')
plt.title('Interpolated x_HI vs. Star Formation Rate Density')
plt.savefig('sfrd_HI.pdf')


