import pandas as pd
import numpy as np
import plotly.graph_objects as go
import xarray as xr
import os
import re
import importlib as il

from tqdm import tqdm
from datetime import datetime, timedelta
import time
import sys
sys.path.append('/glade/work/prasoonv/hpc_course/hw4/')
sys.path.append('/glade/work/prasoonv/hpc_course/hw4/src/')



def lorentzian_histogram(n, bins=100, xmin=-10, xmax=10):
    """
    Sample n random points from the Lorentzian distribution
    using inverse transform sampling. Make a histogram with
    the specified bin count and range. Returns counts.
    """
    u = np.random.random(n) # Uniform(0,1)
    x = 1. / np.tan(np.pi * u) # x = 1/tan(pi*u)
    counts, _ = np.histogram(x, bins=bins, range=(xmin, xmax))
    
    return np.array(counts) # No need to return bin edges for uniform bins


############ HELPER FUNCTIONS ################


def strong_scaling(func, n = 10**7, bins = 100, technique=None):
    
    times = []
    parallel_count = [1, 2, 4, 8, 16, 32]
    
    for p in tqdm(parallel_count):
        
        start = time.perf_counter()
        func(n, n_counts = p, bins = bins)
        end = time.perf_counter() - start
        times.append(end)
    
    fig = go.Figure()
    fig.add_scatter(x=parallel_count, y=times, opacity=0.6)
    fig.update_layout(
        title=f"Strong Scaling: {technique} for n={n:.0e}, bins={bins}",
        xaxis_title="Parallel Divisions",
        yaxis_title="Time (seconds)",
        template="plotly_dark"
    )
    fig.update_xaxes(
        tickvals=parallel_count,
    )
    fig.show()
    fig.write_html(f"{technique}_strong_scaling.html")
    
    return times

def weak_scaling(func, n_per_div = 10**7, bins = 100, technique=None):
    
    times = []
    parallel_count = [1, 2, 4, 8]
    
    for p in tqdm(parallel_count):
        
        n = n_per_div * p
        
        start = time.perf_counter()
        func(n, n_counts = p, bins = bins)
        end = time.perf_counter() - start
        times.append(end)
    
    fig = go.Figure()
    fig.add_scatter(x=parallel_count, y=times, opacity=0.6)
    fig.update_layout(
        title=f"Weak Scaling: {technique} for n per divisions={n_per_div:.0e}, bins={bins}",
        xaxis_title="Parallel Divisions",
        yaxis_title="Time (seconds)",
        template="plotly_dark",
    )
    fig.update_xaxes(
        tickvals=parallel_count,
    )
    #fig.show()
    fig.write_html(f"{technique}_weak_scaling.html")
    
    return times


def speedup(times, technique=None):
    speedups = times[0] / np.array(times)
    
    fig = go.Figure()
    fig.add_scatter(x=[1, 2, 4, 8, 16, 32][:len(times)], y=speedups, opacity=0.6)
    fig.update_layout(
        title=f"Speedup vs Parallel Divisions",
        xaxis_title="Parallel Divisions",
        yaxis_title="Speedup",
        template="plotly_dark"
    )
    fig.update_xaxes(
        tickvals=[1, 2, 4, 8, 16, 32][:len(times)],
    )
    #fig.show()
    fig.write_html(f"{technique}_speedup.html")
    return speedups
    
def efficiency(speedups, parallel_count, technique=None):
    efficiencies = speedups / np.array(parallel_count)
    
    fig = go.Figure()
    fig.add_scatter(x=parallel_count[:len(efficiencies)], y=efficiencies, opacity=0.6)
    fig.update_layout(
        title=f"Efficiency vs Parallel Divisions",
        xaxis_title="Parallel Divisions",
        yaxis_title="Efficiency",
        template="plotly_dark"
    )
    fig.update_xaxes(
        tickvals=parallel_count[:len(efficiencies)],
    )
    #fig.show()
    fig.write_html(f"{technique}_efficiency.html")
    
    return efficiencies

