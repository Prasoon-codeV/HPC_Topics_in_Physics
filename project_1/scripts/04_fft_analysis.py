import xarray as xr
import numpy as np
import time
import os
import glob
import warnings
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# --- CONFIGURATION ---
WACCMX_DIR = '/Users/prasoonv/hpc_project_final/project_1/data/raw/waccmx_data/'
TMP_DIR = '/Users/prasoonv/hpc_project_final/project_1/data/raw/waccmx_tmp/'
OUTPUT_FILE = '/Users/prasoonv/hpc_project_final/project_1/data/processed/waccmx_data/waccmx_fft_2019.nc'

START_DATE = '2019-08-01'
END_DATE = '2019-10-15'

LAT_MIN, LAT_MAX = -40, 20
TARGET_ALT = 110.0  # E-region altitude

os.makedirs(TMP_DIR, exist_ok=True)

def process_single_file(filepath):
    filename = os.path.basename(filepath)
    tmp_path = os.path.join(TMP_DIR, filename)
    
    if os.path.exists(tmp_path):
        return True

    try:
        ds = xr.open_dataset(filepath, engine='netcdf4')
        
        # 1. Drop problematic boundary variables
        bad_vars = ['time_bnds', 'date', 'datesec', 'date_written', 'time_written']
        ds = ds.drop_vars([v for v in bad_vars if v in ds.variables], errors='ignore')
        
        # 2. Standardize dimension names
        rename_map = {"time": "Time", "lat": "GLAT", "lon": "GLON", "ElecColDens": "TEC"}
        ds = ds.rename({k: v for k, v in rename_map.items() if (k in ds.dims or k in ds.variables)})
            
        # 3. Spatial Slicing (Keeping ALL longitudes for 2D FFT)
        if 'GLAT' in ds.dims:
            ds = ds.sel(GLAT=slice(LAT_MIN, LAT_MAX))

        # 4. Altitude conversion and masking for E-region
        if 'Z3' in ds.data_vars and 'lev' in ds.dims:
            r_km = 6371.0
            z3_km = ds["Z3"] * 1e-3
            alt_km = (r_km * z3_km) / (r_km - z3_km)
            
            lo, hi = TARGET_ALT - 5, TARGET_ALT + 5
            mask = (alt_km >= lo) & (alt_km <= hi)
            
            if 'U' in ds.data_vars:
                ds['U_110km'] = ds['U'].where(mask).mean(dim='lev', skipna=True)
            if 'V' in ds.data_vars:
                ds['V_110km'] = ds['V'].where(mask).mean(dim='lev', skipna=True)
                
            ds = ds.drop_vars(['U', 'V', 'Z3', 'lev', 'hyam', 'hybm', 'P0', 'PS'], errors='ignore')
            
        # 5. Keep only final required variables
        keep_vars = ['TEC', 'U_110km', 'V_110km']
        ds = ds[[v for v in keep_vars if v in ds.data_vars]]
        
        ds.to_netcdf(tmp_path)
        ds.close()
        return True
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return False

def clean_tmp_files(file_paths):
    """Helper to delete specific temporary files for benchmarking."""
    for f in file_paths:
        tmp_path = os.path.join(TMP_DIR, os.path.basename(f))
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def benchmark_pipeline(target_files, num_test=8, cpus=8):
    """Runs a strict comparison between serial and multiprocessing execution."""
    print(f"\n--- Running Map-Reduce Benchmark ({num_test} files) ---")
    test_files = target_files[:num_test]
    
    # Ensure a clean slate before testing
    clean_tmp_files(test_files)

    # 1. Serial Execution
    print("Testing Serial Execution...")
    t0 = time.time()
    for f in test_files:
        process_single_file(f)
    serial_time = time.time() - t0
    print(f"Serial Time: {serial_time:.2f}s")

    # Clean slate before parallel test so it doesn't skip
    clean_tmp_files(test_files)

    # 2. Parallel Execution (Multiprocessing)
    print(f"Testing Parallel Execution ({cpus} cores)...")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=cpus) as executor:
        list(executor.map(process_single_file, test_files))
    parallel_time = time.time() - t0
    print(f"Parallel Time: {parallel_time:.2f}s")

    # Calculate and display speedup
    speedup = serial_time / parallel_time if parallel_time > 0 else 0
    print(f"Achieved Speedup: {speedup:.2f}x\n")

def run_pipeline():
    print("--- Starting 3D WACCM-X Map-Reduce Pipeline ---")
    t0 = time.time()

    search_path = os.path.join(WACCMX_DIR, "2019", "*.nc") 
    all_files = sorted(glob.glob(search_path))
    
    target_files = []
    for f in all_files:
        date_str = f.split('.h1.')[-1].replace('-00000.nc', '')
        if START_DATE <= date_str <= END_DATE:
            target_files.append(f)

    print(f"Found {len(target_files)} total files to process.")

    slurm_cpus = 8
    
    # --- RUN THE BENCHMARK FIRST ---
    if len(target_files) >= slurm_cpus:
        benchmark_pipeline(target_files, num_test=8, cpus=slurm_cpus)
    
    print(f"--- Launching Full Pipeline ({slurm_cpus} independent workers) ---")
    with ProcessPoolExecutor(max_workers=slurm_cpus) as executor:
        list(tqdm(executor.map(process_single_file, target_files), total=len(target_files)))

    print("\nMerging temporary files into final 3D dataset...")
    tmp_files = sorted(glob.glob(os.path.join(TMP_DIR, "*.nc")))
    
    if not tmp_files:
        print("No temporary files found. Map step failed.")
        return

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_ds = xr.open_mfdataset(tmp_files, combine='nested', concat_dim='Time')
    
    final_ds = final_ds.sortby('Time')
    _, keep_idx = np.unique(final_ds['Time'].values, return_index=True)
    final_ds = final_ds.isel(Time=np.sort(keep_idx))
    
    comp = dict(zlib=True, complevel=4)
    encoding = {var: comp for var in final_ds.data_vars}
    
    final_ds.to_netcdf(OUTPUT_FILE, encoding=encoding)
    
    print(f"Pipeline Complete in {time.time() - t0:.2f} seconds.")
    print(f"Final output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_pipeline()
