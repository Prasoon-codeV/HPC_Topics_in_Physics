import numpy as np
import cupy as cp
import pandas as pd
import matplotlib.pyplot as plt
import os

from backend import get_backend, to_cpu, Timer


# ------------- Problem setup ------------------------

def setup_problem(d, xp):
    alpha = xp.logspace(0, 6, d)  
    y0 = xp.ones(d)
    return alpha, y0


def exact_solution(alpha, t):
    return np.exp(-alpha * t)


# ---------------- Trapezoidal Rule (TR) --------------------

def solve_TR(alpha, y0, dt, steps, xp):
    y = xp.array(y0, dtype=xp.float64)

    for _ in range(steps):
        factor = (1 - 0.5 * dt * alpha) / (1 + 0.5 * dt * alpha)
        y = factor * y

    return y


# ----------------- TRBDF2 (γ = 2 - sqrt(2)) --------------------

def solve_TRBDF2(alpha, y0, dt, steps, xp):

    gamma = 2 - np.sqrt(2)

    y = xp.array(y0, dtype=xp.float64)

    for _ in range(steps):

        # --- Stage 1: TR (fractional step γ dt)
        a1 = gamma * dt
        factor1 = (1 - 0.5 * a1 * alpha) / (1 + 0.5 * a1 * alpha)
        y_star = factor1 * y

        # --- Stage 2: BDF2-like step
        a2 = (1 - gamma) * dt
        factor2 = 1 / (1 + a2 * alpha)
        y = factor2 * (y_star + (1 - gamma) * y_star)

    return y


# ------------------- Explicit instability demo ---------------

def demo_explicit_instability():

    print("\n=== Explicit instability demo ===")

    alpha = 1e6
    y = 1.0
    dt = 1e-3   # too large for stability

    ys = []

    for _ in range(100):
        y = y - dt * alpha * y  # Forward Euler
        ys.append(y)

    plt.figure()
    plt.plot(ys)
    plt.title("Explicit Euler Instability (α=1e6)")
    plt.xlabel("Step")
    plt.ylabel("y")
    plt.savefig("problem_4/prob4_explicit_instability.png", dpi=200)


# ------------- L-stability comparison -----------------------

def demo_L_stability():

    print("\n=== L-stability comparison ===")

    alphas = np.array([1, 1e3, 1e6])
    y0 = np.ones_like(alphas)

    dt = 0.1
    steps = 50

    y_tr = solve_TR(alphas, y0, dt, steps, np)
    y_trbdf2 = solve_TRBDF2(alphas, y0, dt, steps, np)

    plt.figure()
    plt.plot(alphas, np.abs(y_tr), 'o-', label="TR")
    plt.plot(alphas, np.abs(y_trbdf2), 's-', label="TRBDF2")

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("alpha")
    plt.ylabel("|y| after integration")
    plt.title("L-stability comparison")
    plt.legend()

    plt.savefig("problem_4/prob4_L_stability.png", dpi=200)


# ------------------- GPU Benchmark (TRBDF2) ------------------------

def run_gpu_benchmark():

    print("\n=== GPU Benchmark (TRBDF2) ===")

    d = int(1e6)
    dt = 1e-3
    T = 1
    steps = int(T / dt)

    backend = get_backend(prefer_gpu=True)

    alpha_cpu, y0_cpu = setup_problem(d, np)

    # CPU
    with Timer(get_backend(False)) as t:
        y_cpu = solve_TRBDF2(alpha_cpu, y0_cpu, dt, steps, np)
    cpu_time = t.dt

    # GPU
    alpha_gpu, y0_gpu = setup_problem(d, backend.xp)

    # Warmup
    if backend.has_gpu:
        solve_TRBDF2(alpha_gpu[:1000], y0_gpu[:1000], dt, 10, backend.xp)

    with Timer(backend) as t:
        y_gpu = solve_TRBDF2(alpha_gpu, y0_gpu, dt, steps, backend.xp)
    gpu_time = t.dt

    y_gpu = to_cpu(y_gpu)

    speedup = cpu_time / gpu_time

    print(f"CPU time: {cpu_time:.4f}")
    print(f"GPU time: {gpu_time:.4f}")
    print(f"Speedup: {speedup:.2f}")

    return cpu_time, gpu_time, speedup


# ------------- Summary table ------------------------

def generate_summary():

    print("\n=== Summary Table ===")

    rows = [
        ["TR", "Not L-stable", "Moderate", "Poor damping of stiff modes"],
        ["TRBDF2", "L-stable", "High", "Strong damping, better for stiff systems"]
    ]

    df = pd.DataFrame(rows, columns=[
        "Method",
        "Stability",
        "Accuracy",
        "Notes"
    ])

    print(df)

    df.to_csv("problem_4/prob4_summary.csv", index=False)


# ---------------- Problem 4 ------------------------------

if __name__ == "__main__":


    demo_explicit_instability()
    demo_L_stability()

    run_gpu_benchmark()

    generate_summary()

