import os
import re
import time
import argparse
import pandas as pd
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# =========================
# CONFIG
# =========================
BASE_DIR = Path('/Users/prasoonv/hpc_project_final/project_1/data/raw/lisn_data/')
OUTPUT_DIR = Path('/Users/prasoonv/hpc_project_final/project_1/data/processed/lisn_data/')
MIN_ELEV = 30.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# HELPERS
# =========================
def _date_from_filename(name: str):
    m = re.search(r"(\d{2})(\d{3})\.dat$", name.lower())
    if not m:
        return None
    yy, doy = int(m.group(1)), int(m.group(2))
    year = 2000 + yy
    return (pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)).date()

def _parse_one_file(args):
    """Parses a single .dat file and extracts VTEC data."""
    fp_str, min_elev = args
    fp = Path(fp_str)

    file_date = None
    fallback_date = _date_from_filename(fp.name)
    local_daily = defaultdict(list)

    with fp.open("r", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue

            if (file_date is None and len(parts) == 3 and 
                all(p.lstrip("+-").isdigit() for p in parts) and len(parts[0]) == 4):
                try:
                    y, m, d = map(int, parts)
                    file_date = pd.Timestamp(year=y, month=m, day=d).date()
                except Exception:
                    pass
                continue

            if len(parts) < 11:
                continue

            try:
                hour, minute, second = int(parts[2]), int(parts[3]), int(parts[4])
                vtec = float(parts[5])
                elev = float(parts[7])
                ipp_lat = float(parts[8])
                ipp_lon = float(parts[9])
            except Exception:
                continue

            if elev < min_elev:
                continue

            base_date = file_date if file_date else fallback_date
            if base_date is None:
                continue

            dt = pd.Timestamp(base_date) + pd.Timedelta(hours=hour, minutes=minute, seconds=second)
            day_key = dt.date().isoformat()

            local_daily[day_key].append({
                "date": dt,
                "vtec": vtec,
                "ipp_lat": ipp_lat,
                "ipp_lon": ipp_lon,
            })

    return local_daily

# =========================
# BENCHMARKING
# =========================
def run_serialization_benchmark():
    """Benchmarks CSV vs Parquet writing on a sample of data."""
    print("--- Running Serialization Benchmark (CSV vs Parquet) ---")
    
    # 1. Gather a decent amount of data to test
    day_dirs = sorted(BASE_DIR.glob('day140-149'))
    if not day_dirs:
        print("No data found to benchmark. Please check BASE_DIR.")
        return
        
    test_files = sorted(day_dirs[0].glob("*.dat"))[:20] 
    print(f"Parsing {len(test_files)} files to generate test DataFrame...")
    
    daily_rows = defaultdict(list)
    for fp in test_files:
        res = _parse_one_file((str(fp), MIN_ELEV))
        for day_key, rows in res.items():
            daily_rows[day_key].extend(rows)
            
    # Combine into one big DataFrame for the test
    all_rows = []
    for rows in daily_rows.values():
        all_rows.extend(rows)
        
    df = pd.DataFrame(all_rows, columns=["date", "vtec", "ipp_lat", "ipp_lon"])
    
    # --- DOWNCASTING (HPC Memory Technique) ---
    df = df.astype({"vtec": "float32", "ipp_lat": "float32", "ipp_lon": "float32"})
    
    print(f"\nDataFrame created with {len(df)} rows.")

    # 2. Benchmark CSV
    csv_path = OUTPUT_DIR / "benchmark_test.csv"
    t0 = time.time()
    df.to_csv(csv_path, index=False)
    csv_time = time.time() - t0
    csv_size = os.path.getsize(csv_path) / (1024 * 1024)

    # 3. Benchmark Parquet
    pq_path = OUTPUT_DIR / "benchmark_test.parquet"
    t0 = time.time()
    df.to_parquet(pq_path, engine='pyarrow', compression='snappy')
    pq_time = time.time() - t0
    pq_size = os.path.getsize(pq_path) / (1024 * 1024)

    # 4. Results
    print("\n=========================================")
    print(f"| Metric      | CSV       | Parquet   |")
    print("|-------------|-----------|-----------|")
    print(f"| Write Time  | {csv_time:.2f}s     | {pq_time:.2f}s     |")
    print(f"| File Size   | {csv_size:.2f} MB | {pq_size:.2f} MB  |")
    print("=========================================")
    print(f"Write Speedup:  {csv_time/pq_time:.2f}x")
    print(f"Storage Saved:  {(1 - pq_size/csv_size)*100:.1f}%\n")

    # Cleanup
    csv_path.unlink()
    pq_path.unlink()


# =========================
# MAIN PROCESSING
# =========================
def process_all_folders():
    day_dirs = sorted(BASE_DIR.glob('day140-149'))

    if not day_dirs:
        print("No dayXXX-XXX folders found.")
        return

    ctx = mp.get_context("fork")

    for day_dir in day_dirs:
        print(f"\nProcessing folder: {day_dir.name}")

        files = sorted(day_dir.glob("*.dat"))
        if not files:
            continue

        out_subdir = OUTPUT_DIR / day_dir.name
        out_subdir.mkdir(parents=True, exist_ok=True)

        if any(out_subdir.glob("*.parquet")):
            print(f"Skipping {day_dir.name} (already processed)")
            continue

        daily_rows = defaultdict(list)
        max_workers = min(8, os.cpu_count() or 1, len(files))

        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
            tasks = ((str(fp), MIN_ELEV) for fp in files)
            for local_daily in tqdm(ex.map(_parse_one_file, tasks, chunksize=8), total=len(files), desc=day_dir.name):
                for day_key, rows in local_daily.items():
                    daily_rows[day_key].extend(rows)

        # Write Parquet per day
        for day_key, rows in sorted(daily_rows.items()):
            out_df = pd.DataFrame(rows, columns=["date", "vtec", "ipp_lat", "ipp_lon"])
            
            # --- Precision Downcasting ---
            out_df = out_df.astype({"vtec": "float32", "ipp_lat": "float32", "ipp_lon": "float32"})
            
            out_path = out_subdir / f"tec_{day_key.replace('-', '')}.parquet"
            out_df.to_parquet(out_path, engine='pyarrow', compression='snappy')

        print(f"Finished {day_dir.name} to {out_subdir}")

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel LISN Parser and Serializer")
    parser.add_argument("--benchmark", action="store_true", help="Run a timing and sizing benchmark for CSV vs Parquet")
    args = parser.parse_args()

    if args.benchmark:
        run_serialization_benchmark()
    else:
        process_all_folders()
        
        
        