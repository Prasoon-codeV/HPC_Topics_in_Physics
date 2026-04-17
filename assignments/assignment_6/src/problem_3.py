import numpy as np
import cupy as cp
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

from backend import get_backend, to_cpu, Timer


# ------------  Logistic equation: y' = r y (1 - y) -------------

def logistic_rhs(y, r=2.0):
    return r * y * (1 - y)


# ---------  Batched RK4 solver (vectorized) -----------------

def rk4_batch(y0, dt, steps, xp):

    y = xp.array(y0, dtype=xp.float64)

    for _ in range(steps):
        k1 = logistic_rhs(y)
        k2 = logistic_rhs(y + 0.5*dt*k1)
        k3 = logistic_rhs(y + 0.5*dt*k2)
        k4 = logistic_rhs(y + dt*k3)

        y = y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

    return y


# ------------- Benchmark function ----------------------------

def run_benchmark():

    print("\n=== Problem 3: GPU Batch Benchmark ===")

    # Parameters
    r = 2.0
    dt = 1e-3
    T = 10
    steps = int(T / dt)

    # Ensemble sizes
    Ns = [10**3, 10**4, 10**5, 10**6]

    backend = get_backend(prefer_gpu=True)

    # Warm-up GPU (IMPORTANT)
    if backend.has_gpu:
        y_warm = np.linspace(0.01, 0.99, 1000)
        rk4_batch(backend.xp.array(y_warm), dt, 10, backend.xp)
        backend.xp.cuda.Stream.null.synchronize()

    rows = []

    for N in Ns:

        print(f"\nRunning N = {N}")

        y0 = np.linspace(0.01, 0.99, N)

        # ---------------- CPU ----------------
        with Timer(get_backend(False)) as t:
            rk4_batch(y0, dt, steps, np)
        cpu_time = t.dt

        # ---------------- GPU ----------------
        with Timer(backend) as t:
            y0_gpu = backend.xp.array(y0)
            rk4_batch(y0_gpu, dt, steps, backend.xp)
        gpu_time = t.dt

        # ---------------- Metrics ----------------
        speedup = cpu_time / gpu_time if gpu_time > 0 else np.nan

        throughput_cpu = (N * steps) / cpu_time
        throughput_gpu = (N * steps) / gpu_time

        rows.append([
            N,
            np.log10(N),
            cpu_time,
            gpu_time,
            speedup,
            throughput_cpu,
            throughput_gpu
        ])

    df = pd.DataFrame(rows, columns=[
        "N",
        "log10(N)",
        "CPU Time (s)",
        "GPU Time (s)",
        "Speedup",
        "CPU Throughput",
        "GPU Throughput"
    ])

    df.to_csv("problem_3_fin/prob3_results.csv", index=False)

    print("\nResults:")
    print(df)

    return df


# ----------------- Plotting -----------------------------

def plot_results(df):

    # --- Runtime ---
    plt.figure()
    plt.plot(df["log10(N)"], df["CPU Time (s)"], marker='o', label="CPU")
    plt.plot(df["log10(N)"], df["GPU Time (s)"], marker='s', label="GPU")
    plt.xlabel("log10(N)")
    plt.ylabel("Time (s)")
    plt.title("Runtime vs Problem Size")
    plt.legend()
    plt.grid()
    plt.savefig("problem_3_fin/prob3_runtime.png", dpi=200)

    # --- Speedup ---
    plt.figure()
    plt.plot(df["log10(N)"], df["Speedup"], marker='o')
    plt.xlabel("log10(N)")
    plt.ylabel("Speedup (CPU/GPU)")
    plt.title("GPU Speedup")
    plt.grid()
    plt.savefig("problem_3_fin/prob3_speedup.png", dpi=200)

    # --- Throughput ---
    plt.figure()
    plt.plot(df["log10(N)"], df["CPU Throughput"], marker='o', label="CPU")
    plt.plot(df["log10(N)"], df["GPU Throughput"], marker='s', label="GPU")
    plt.xlabel("log10(N)")
    plt.ylabel("Trajectories x Steps / second")
    plt.title("Throughput")
    plt.legend()
    plt.grid()
    plt.savefig("problem_3_fin/prob3_throughput.png", dpi=200)

    plt.show()


# ------------------ Problem 3 ----------------------------------

if __name__ == "__main__":

    df = run_benchmark()
    plot_results(df)



