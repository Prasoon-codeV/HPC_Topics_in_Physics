import os
import time
import glob
import pandas as pd
import numpy as np
import xarray as xr
import dask.dataframe as dd
from scipy.ndimage import gaussian_filter
from joblib import Parallel, delayed, cpu_count
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import datetime as dt
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
START_DATE = dt.datetime(2019, 8, 15)
END_DATE = dt.datetime(2019, 10, 31)
LAT_RANGE = (-40, 20)
LON_RANGE = (-75, -65)
TARGET_UT = 21

# Paths (Adjust these to your local structure)
BASE_DIR = "/Users/prasoonv/hpc_project_final/project_1/"
LISN_DIR = f"{BASE_DIR}data/raw/lisn_data/"
WACCMX_FILE = f"{BASE_DIR}data/processed/waccmx_data/waccmx_fft_2019.nc"


# ==========================================
# 2. COMPUTE BENCHMARKING (SPATIAL & GAUSSIAN)
# ==========================================
def process_single_day(date, df_subset):
    """CPU-bound task: 2D Histogram binning and Gaussian smoothing for a single day."""
    if df_subset.empty:
        return date, np.full((60,), np.nan) # Assuming 1-degree lat bins
        
    lons = df_subset["GLON"].to_numpy()
    lats = df_subset["GDLAT"].to_numpy()
    vtec_vals = df_subset["TEC"].to_numpy()

    # Grid parameters
    lon_edges = np.linspace(-75, -65, 11) # 1-degree resolution
    lat_edges = np.linspace(-40, 20, 61)  # 1-degree resolution

    sum_vtec, _, _ = np.histogram2d(lats, lons, bins=[lat_edges, lon_edges], weights=vtec_vals)
    count, _, _ = np.histogram2d(lats, lons, bins=[lat_edges, lon_edges])

    mean_vtec = np.divide(sum_vtec, count, out=np.full_like(sum_vtec, np.nan), where=count > 0)

    # Gaussian Smoothing (NaN safe)
    valid_mask = ~np.isnan(mean_vtec)
    filled = np.where(valid_mask, mean_vtec, 0)
    
    smooth = gaussian_filter(filled, sigma=1.2)
    weights = gaussian_filter(valid_mask.astype(float), sigma=1.2)
    
    mean_vtec_smooth = np.divide(smooth, weights, out=np.full_like(smooth, np.nan), where=weights > 0)

    # Average across the longitude band (axis 1) to get a 1D array of Latitude vs VTEC for this day
    zonal_mean = np.nanmean(mean_vtec_smooth, axis=1)
    return date, zonal_mean


# ==========================================
# 2. DASK LOADING & FILTERING FOR LISN
# ==========================================
def load_lisn_with_dask(base_dir, lat_range, lon_range, target_ut):
    """
    Lazily loads LISN Parquet files across all subfolders, applies spatial/temporal 
    filtering, and materializes the result into a Pandas DataFrame.
    """
    print("--- Loading LISN Parquet Data with Dask ---")
    t0 = time.time()
    
    # Dask handles the globbing across all dayXXX-XXX subfolders automatically
    file_pattern = os.path.join(base_dir, "data/raw/lisn_data/*/*.parquet")
    
    # Read binary files using PyArrow engine
    ddf = dd.read_parquet(file_pattern, engine='pyarrow')
    
    # Rename columns so it works seamlessly with benchmark_compute()
    ddf = ddf.rename(columns={
        'ipp_lat': 'GDLAT',
        'ipp_lon': 'GLON',
        'vtec': 'TEC',
        'date': 'DT'
    })
    
    # 1. Spatial Filter (Pushing this to Dask before compute() saves massive memory)
    ddf = ddf[
        (ddf['GDLAT'] >= lat_range[0]) & (ddf['GDLAT'] <= lat_range[1]) &
        (ddf['GLON'] >= lon_range[0]) & (ddf['GLON'] <= lon_range[1])
    ]
    
    # 2. Temporal Filter (~21 UT +/- 30 mins)
    ddf = ddf[
        (ddf['DT'].dt.hour == target_ut) | 
        ((ddf['DT'].dt.hour == target_ut - 1) & (ddf['DT'].dt.minute >= 30))
    ]
    
    # Materialize the distributed graph into a single Pandas DataFrame
    df = ddf.compute()
    
    load_time = time.time() - t0
    print(f"LISN Dask Load & Filter Time: {load_time:.2f}s")
    print(f"Total LISN rows retained for compute: {len(df)}\n")
    
    return df

def load_lisn_with_dask(parquet_dir, lat_range, lon_range, target_ut):
    """
    Lazily loads LISN Parquet files, applies spatial/temporal 
    filtering, and materializes the result into a Pandas DataFrame.
    """
    print("--- Loading Actual LISN Parquet Data with Dask ---")
    t0 = time.time()
    
    # Read all binary parquet files directly from the outputs folder
    file_pattern = os.path.join(parquet_dir, "day*/*.parquet")
    ddf = dd.read_parquet(file_pattern, engine='pyarrow')
    
    #print(ddf.head())  # Debug: Check initial columns and data types
    
    # Rename columns so it works seamlessly with benchmark_compute()
    ddf = ddf.rename(columns={
        'ipp_lat': 'GDLAT',
        'ipp_lon': 'GLON',
        'vtec': 'TEC',
        'date': 'DT'
    })
    
    # 1. Spatial Filter (Pushing this to Dask before compute() saves massive memory)
    ddf = ddf[
        (ddf['GDLAT'] >= lat_range[0]) & (ddf['GDLAT'] <= lat_range[1]) &
        (ddf['GLON'] >= lon_range[0]) & (ddf['GLON'] <= lon_range[1])
    ]
    
    # 2. Temporal Filter (~21 UT +/- 30 mins)
    ddf = ddf[
        (ddf['DT'].dt.hour == target_ut) | 
        ((ddf['DT'].dt.hour == target_ut - 1) & (ddf['DT'].dt.minute >= 30))
    ]
    
    # Materialize the distributed graph into a single Pandas DataFrame
    df = ddf.compute()
    
    load_time = time.time() - t0
    print(f"LISN Dask Load & Filter Time: {load_time:.2f}s")
    print(f"Total LISN rows retained for compute: {len(df)}\n")
    
    return df


def benchmark_compute(df, dataset_name="Dataset"):
    """Benchmarks serial vs process-parallel vs thread-parallel execution."""
    print(f"--- Running Compute Benchmark for {dataset_name} ---")
    
    # 1. Standardize the Date column
    if 'DT' not in df.columns:
        df['DT'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])
    else:
        df['DT'] = pd.to_datetime(df['DT'])
        
    df['Date_Only'] = df['DT'].dt.floor('D')
    grouped = [group for name, group in df.groupby('Date_Only')]
    
    print(f"Total days to process: {len(grouped)} | Total rows: {len(df)}")
    
    # ---------------------------------------------------------
    # TEST 1: SERIAL BASELINE
    # ---------------------------------------------------------
    t0 = time.time()
    serial_res = [process_single_day(group['Date_Only'].iloc[0], group) for group in grouped]
    serial_time = time.time() - t0
    print(f"1. Serial Loop: {serial_time:.2f}s")

    # ---------------------------------------------------------
    # TEST 2: MULTIPROCESSING (Distributed Memory - High Overhead)
    # ---------------------------------------------------------
    t0 = time.time()
    slurm_cpus = 6 #int(os.environ.get('SLURM_CPUS_PER_TASK', (os.cpu_count() - 2) or 1))
    
    _ = Parallel(n_jobs=slurm_cpus)(
        delayed(process_single_day)(group['Date_Only'].iloc[0], group) for group in grouped
    )
    process_time = time.time() - t0
    print(f"2. Joblib Processes ({slurm_cpus} cores): {process_time:.2f}s (Speedup: {serial_time/process_time:.2f}x)")

    # ---------------------------------------------------------
    # TEST 3: MULTITHREADING (Shared Memory - Low Overhead)
    # ---------------------------------------------------------
    t0 = time.time()
    
    thread_res = Parallel(n_jobs=slurm_cpus, prefer="threads")(
        delayed(process_single_day)(group['Date_Only'].iloc[0], group) for group in grouped
    )
    thread_time = time.time() - t0
    print(f"3. Joblib Threads ({slurm_cpus} threads): {thread_time:.2f}s (Speedup: {serial_time/thread_time:.2f}x)\n")
    
    # Restructure into Time x Lat array using the fastest result
    thread_res.sort(key=lambda x: x[0]) 
    dates = [x[0] for x in thread_res]
    vtec_matrix = np.vstack([x[1] for x in thread_res])
    
    return dates, vtec_matrix


# ==========================================
# 4. PLOTTING PIPELINE (4-Row Comparison)
# ==========================================
def plot_4_rows(waccmx_ds, lisn_data, date_range, target_ut=21):
    """
    Constructs a 4-row stacked plot sharing the time axis.
    Rows: LISN VTEC, WACCM-X TEC, WACCM-X V_110km, WACCM-X U_110km.
    """
    start_date, end_date = date_range
    
    # Adjust height to 24 for 4 rows
    fig = plt.figure(figsize=(14, 24)) 
    gs = gridspec.GridSpec(
        4, 2, 
        width_ratios=[20, 1], 
        height_ratios=[1, 1, 1, 1], 
        hspace=0.20, wspace=0.05
    )

    # HELPER FOR CONTOUR PLOTS 
    # Added dynamic cmap, levels, and zlabel to handle both TEC and Winds
    def plot_contour(ax_idx, data_matrix, dates, lats, title, zlabel, levels, cmap, extend='max'):
        ax = fig.add_subplot(gs[ax_idx, 0])
        cax = fig.add_subplot(gs[ax_idx, 1])
        
        # Convert dates to numerical format for contourf
        T, LAT = np.meshgrid(mdates.date2num(dates), lats)
        
        # Ensure data matrix matches (Lat, Time) shape for plotting
        if data_matrix.shape == (len(dates), len(lats)):
            plot_data = data_matrix.T
        else:
            plot_data = data_matrix
            
        cf = ax.contourf(T, LAT, plot_data, levels=levels, cmap=cmap, extend=extend)
        cbar = fig.colorbar(cf, cax=cax)
        cbar.set_label(zlabel, fontsize=16)
        cbar.ax.tick_params(labelsize=14)
        
        ax.set_title(title, fontsize=20)
        ax.set_ylabel("Latitude", fontsize=18)
        ax.set_xlim(start_date, end_date)
        ax.set_ylim(-40, 20)
        ax.grid(True, alpha=0.3)
        
        # Formatting Time Axis
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.tick_params(axis='both', labelsize=16)
        
        # Specific Line Markers
        ax.axhline(-11, color='black', linestyle='--', linewidth=1.2, label='Magnetic Equator')
        ax.axvline(mdates.date2num(dt.datetime(2019, 9, 1)), color='magenta', linestyle='--', linewidth=1.2, label='SSW Onset')
        
        # Hide x-tick labels for top 3 rows to keep the plot clean
        if ax_idx < 3:
            ax.set_xticklabels([])
            
        return ax

    lat_grid = np.linspace(-40, 20, 60) # Must match your binning length
    waccmx_ut = waccmx_ds.where((waccmx_ds.Time.dt.hour == target_ut), drop=True)

    # ---------------------------------------------------------
    # ROW 1: LISN VTEC
    # ---------------------------------------------------------
    l_dates, l_matrix = lisn_data
    plot_contour(0, l_matrix, l_dates, lat_grid, 
                 f"LISN VTEC (Smoothed, 65-75W) at UT={target_ut}H", 
                 "VTEC (TECU)", np.arange(0, 32, 2), 'jet')

    # ---------------------------------------------------------
    # ROW 2: WACCM-X TEC
    # ---------------------------------------------------------
    plot_contour(1, waccmx_ut['TEC'].values, waccmx_ut['Time'].values, waccmx_ut['GLAT'].values, 
                 f"WACCM-X TEC along GLON=75W at UT={target_ut}H", 
                 "TEC (TECU)", np.arange(0, 32, 2), 'jet')

    # ---------------------------------------------------------
    # ROW 3: WACCM-X V_110km (Meridional Wind / Tides)
    # ---------------------------------------------------------
    # Using RdBu_r for winds (Red = Northward, Blue = Southward)
    plot_contour(2, waccmx_ut['V_110km'].values, waccmx_ut['Time'].values, waccmx_ut['GLAT'].values, 
                 f"WACCM-X Meridional Wind (V) at 110km, UT={target_ut}H", 
                 "V (m/s)", np.linspace(-60, 60, 21), 'RdBu_r', extend='both')

    # ---------------------------------------------------------
    # ROW 4: WACCM-X U_110km (Zonal Wind / Planetary Waves)
    # ---------------------------------------------------------
    # (Red = Eastward, Blue = Westward)
    plot_contour(3, waccmx_ut['U_110km'].values, waccmx_ut['Time'].values, waccmx_ut['GLAT'].values, 
                 f"WACCM-X Zonal Wind (U) at 110km, UT={target_ut}H", 
                 "U (m/s)", np.linspace(-60, 60, 21), 'RdBu_r', extend='both')

    plt.tight_layout()
    plt.savefig("ssw_ionosphere_winds_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Define bounds
    DATE_RANGE = (dt.datetime(2019, 8, 1), dt.datetime(2019, 10, 15))
    TARGET_UT = 21
    
    # 1. FFT Processed WACCM-X loading and processing
    waccmx_file = WACCMX_FILE
    ds_waccmx = xr.open_dataset(waccmx_file)
    #ds_waccmx = ds_waccmx.rename({"GLAT": "GDLAT"})
    
    # 2. LISN data loading and processing (with benchmarking)
    lisn_raw = load_lisn_with_dask(LISN_DIR, LAT_RANGE, LON_RANGE, TARGET_UT)
    lisn_processed = benchmark_compute(lisn_raw, dataset_name="LISN")
    
    # 3. Plotting
    plot_4_rows(ds_waccmx, lisn_processed, DATE_RANGE, target_ut=TARGET_UT)
