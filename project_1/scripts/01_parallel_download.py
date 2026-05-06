import os
import glob
import time
import shutil
import argparse
import subprocess
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION ---
FULL_NAME = "Prasoon"
EMAIL = "pxv220016@utdallas.edu"
AFFILIATION = "UTDallas" 
OUTDIR = "/Users/prasoonv/hpc_project_final/project_1/data/raw/madrigal_data/"

def generate_tasks(start="08/15/2019", end="10/02/2019", step_days=1):
    """Generates time chunks with custom logic to bypass overlapping days."""
    start_dt = datetime.strptime(start, '%m/%d/%Y')
    end_dt = datetime.strptime(end, '%m/%d/%Y')
    delta = timedelta(days=step_days)

    tasks = []
    t = start_dt

    while t <= end_dt:
        t2 = min(t, end_dt) # + delta - timedelta(days=1), end_dt)
        tasks.append((t.strftime("%m/%d/%Y"), t2.strftime("%m/%d/%Y")))
        t += delta + timedelta(days=1)

    return tasks

def process_chunk(task):
    """Downloads and unzips a single chunk in a thread-safe temporary directory."""
    start_date, end_date = task
    chunk_name = f"tmp_{start_date.replace('/','')}_{end_date.replace('/','')}"
    chunk_dir = os.path.join(OUTDIR, chunk_name)
    os.makedirs(chunk_dir, exist_ok=True)

    cmd = [
        "globalDownload.py", "--verbose",
        "--url=http://cedar.openmadrigal.org",
        f"--outputDir={chunk_dir}",
        f"--user_fullname={FULL_NAME}",
        f"--user_email={EMAIL}",
        f"--user_affiliation={AFFILIATION}",
        "--format=ascii",
        f"--startDate={start_date}",
        f"--endDate={end_date}",
        "--inst=8000",
        "--kindat=3500"
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)

        gz_files = glob.glob(os.path.join(chunk_dir, "*.gz"))
        for gz in gz_files:
            subprocess.run(["gunzip", "-f", gz], check=True)

        extracted_files = glob.glob(os.path.join(chunk_dir, "*"))
        for f in extracted_files:
            shutil.move(f, os.path.join(OUTDIR, os.path.basename(f)))

        os.rmdir(chunk_dir)
        return True

    except subprocess.CalledProcessError as e:
        print(f"Failed: {start_date} to {end_date}. Error: {e.stderr}")
        return False

def wipe_downloads():
    """Safely clears the output directory for a fresh benchmark."""
    for f in glob.glob(os.path.join(OUTDIR, "*")):
        if os.path.isfile(f):
            os.remove(f)

def run_benchmark():
    """Sweeps through 1, 2, 4, 6, and 8 workers and generates a Markdown table."""
    os.makedirs(OUTDIR, exist_ok=True)
    
    # 12 days starting from August 15, 2019
    test_tasks = generate_tasks(start="09/02/2019", end="09/24/2019", step_days=1)
    
    print(f"--- STARTING BENCHMARK ON {len(test_tasks)} CHUNKS ---")
    results_table = []
    
    # 1. SERIAL RUN (Baseline)
    print("Running Serial Baseline (1 Worker)...")
    wipe_downloads()
    t0 = time.time()
    for task in test_tasks:
        process_chunk(task)
    serial_time = time.time() - t0
    results_table.append((1, serial_time, 1.00))
    print(f"Serial Time: {serial_time:.2f}s\n")

    # 2. PARALLEL RUNS
    worker_counts = [2, 3, 4, 6, 8] #[3, 4] #[2, 3, 4, 6, 8]
    
    for w in worker_counts:
        print(f"Running Parallel Test with {w} Workers...")
        wipe_downloads()  # Ensure fair test
        
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=w) as executor:
            list(executor.map(process_chunk, test_tasks))
        p_time = time.time() - t0
        
        speedup = serial_time / p_time
        results_table.append((w, p_time, speedup))
        print(f"Time: {p_time:.2f}s | Speedup: {speedup:.2f}x\n")

    # 3. PRINT FORMATTED TABLE
    print("=========================================")
    print("| Workers | Time (seconds) | Speedup (x) |")
    print("|---------|----------------|-------------|")
    for row in results_table:
        print(f"| {row[0]:<7} | {row[1]:<14.2f} | {row[2]:<11.2f} |")
    print("=========================================")


def run_full_download():
    """Executes the full downloading pipeline for the whole dataset using 8 workers."""
    os.makedirs(OUTDIR, exist_ok=True)
    tasks = generate_tasks()
    print(f"Total tasks generated: {len(tasks)}")
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_chunk, tasks))
        
    successes = sum(1 for r in results if r is True)
    print(f"Completed {successes}/{len(tasks)} downloads successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Madrigal Downloader")
    parser.add_argument("--benchmark", action="store_true", help="Run a scaling benchmark on 12 chunks")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark()
    else:
        run_full_download()
        
