from numpy.random import rand
import sys
import numpy as np
from tqdm import tqdm
from memory_profiler import memory_usage as mmu
import time

import random, sys
from numba import jit, njit, prange

import xarray as xr
#import plotly.express as py
import pandas as pd

#import plotly.graph_objects as go

# Python code - loop

def calc_pi_loop(n):
    h = 0 # Number of hits inside the circle

    for _ in range(n):
        x, y = rand(), rand() # Random points in [0, 1)

        if x*x + y*y < 1.:
            h += 1 # Successful hit

    return 4. * float(h) / float(n) # Estimate pi

# Numpy code

def calc_pi_numpy(n):

    h = sum(rand(n)**2 + rand(n)**2 < 1.)

    return 4. * float(h) / float(n) # Estimate pi


# Numba codes - normal and parallel one

@njit
def calc_pi_numba(n):
    h = 0
    for _ in range(n):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)

        if x*x + y*y < 1.:
            h += 1

    return 4. * h / n

@jit(nopython=True, nogil=True, parallel=True)
def calc_pi_parallel(n):
    h = 0
    for _ in prange(n):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        if x**2 + y**2 < 1:
            h += 1
    return 4. * h / n


# Selection of the function to run and the number of iterations

def mem_time_calc(n, func, logs=False):

    if func == 'python':
        f = lambda: calc_pi_loop(int(n))

    elif func == 'numpy':
        f = lambda: calc_pi_numpy(int(n))
        
    elif func == 'numba':
        f = lambda: calc_pi_numba(int(n))
        
    elif func == 'numba_parallel':
        f = lambda: calc_pi_parallel(int(n))
    
    
    start = time.perf_counter() 
    max_mem, pi_est = mmu(f, max_usage=True, retval=True)
    elapsed = time.perf_counter() - start
    
    if logs is not None and logs:
        print(f"N = {n:0.0e}, pi = {pi_est:0.4e}, time = {elapsed:0.4e}s, max_memory={max_mem}MiB")
        print()
        
    return pi_est, elapsed, max_mem    




type = ['python', 'numpy', 'numba', 'numba_parallel']

N = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]


df = pd.DataFrame(columns=['method', 'N', 'pi_mean', 'pi_std', 'pi_abs_error', 'runtime', 'time_per_sec', 'memory_used', 'sample_per_sec'])

for t in type:


    for n in tqdm(N):

        pi_n = []
        elapsed_n = []
        max_mem_n = []

        for i in tqdm(range(10)):
            
            if (t != 'numba_parallel'):
                if  (i > 0):
                    break 

            a, b, c = mem_time_calc(n, func = t, logs=False)

            pi_n.append(a)
            elapsed_n.append(b)
            max_mem_n.append(c)

        if len(pi_n) == 0:
            continue
        
        runtime = sum(elapsed_n)/len(elapsed_n)
        time_per_iter = runtime/n
        sample_per_sec = n/runtime
        memory_used = sum(max_mem_n)/len(max_mem_n)

        pi_mean = sum(pi_n)/len(pi_n)
        pi_abs_error = sum([abs(x - pi_mean) for x in pi_n])/len(pi_n)
        pi_std = np.std(pi_n)

        results = {'method':t,
                   'N':n,
                   'pi_mean':pi_mean,
                   'pi_std':pi_std,
                   'pi_abs_error':pi_abs_error,
                   'runtime':runtime,
                   'time_per_sec':time_per_iter,
                   'memory_used':memory_used,
                   'sample_per_sec':sample_per_sec
                   }

        df = pd.concat([df, pd.DataFrame([results])], ignore_index=True)

    print(f'Completed calculation using method: {t}')

df.to_csv('all_pi_estimation_results.csv', index=False)


