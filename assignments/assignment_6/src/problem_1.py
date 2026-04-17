import numpy as np
import cupy as cp
import pandas as pd
import matplotlib.pyplot as plt
import time
import argparse

from backend import get_backend, to_cpu, Timer


# -------------------------------------------------------------------

# Test example ODE
def exp_decay(t, y):
    # Non-stiff ODE example
    # y'(t) = -y, y(0) = 1, for t in [0, 1]
    # solution, y(t) = exp(-t)
    return -y

def exp_decay_solution(t):
    return np.exp(-t)

# --------------------------------------------------------------------

# Main IVP ODE Solver function
def solve_ivp(f, y0, dt, y_lim = (0,1), method = 'euler', xp = np):

    '''f: function defining ODE system, f(t, y)
       y0: initial condition, y(t=0)
       dt: time step size
       y_lim: limits of integration
       method: method to use (eg. - 'euler', 'rk2', 'rk4')
       xp: module, either numpy or cupy for array operations
    '''

    t0, tf = y_lim
    steps = int((tf - t0) / dt)
    t = t0

    # Defining array based on the backend (numpy or cupy)
    y = xp.array(y0, dtype=xp.float64)

    for _ in range(steps):

        # ----------- EULER INTEGRATION Method ---------------
        if method == 'euler':                                                                                                                                                                                                     
            y = y + dt * f(t, y)                                                                                                                                                                                                        
                                                                                                                                                                                                                                     
        # ----------- 2nd-ORDER RUNGE KUTTA Method -----------                                                                                                                                                                       
        elif method == 'rk2':                                                                                                                                                                                                     
            k1 = f(t, y)                                                                                                                                                                                                             
            k2 = f(t + dt/2, y + (dt/2) * k1)                                                                                                                                                                                          
            y = y + dt * k2                                                                                                                                                                                                             
                                                                                                                                                                                                                                     
        # ---------- 4th-ORDER RUNGE KUTTA Method -----------                                                                                                                                                                        
        elif method == 'rk4':                                                                                                                                                                                                     
            k1 = f(t, y)                                                                                                                                                                                                             
            k2 = f(t + dt/2, y + (dt/2) * k1)                                                                                                                                                                                          
            k3 = f(t + dt/2, y + (dt/2) * k2)                                                                                                                                                                                          
            k4 = f(t + dt, y + dt * k3)                                                                                                                                                                                              
            y = y + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)                                                                                                                                                                                  
        t += dt                                                                                                                                                                                                                      
    return y                                                                                                                                                                                                                                                                                                                                                                                                                                
                                                                                                                                                                                                                                     
# ---------------------------------------------------------------------                                                                                                                                                              
                                                                                                                                                                                                                                     
# Validaetion and testing

def run_validation():
    print("\n=== Validation (CPU vs GPU) ===")

    dt = 1e-3

    y_cpu = solve_ivp(exp_decay, 1.0, dt, method="rk4", xp=np)

    backend = get_backend(prefer_gpu=True)
    y_gpu = solve_ivp(exp_decay, 1.0, dt, method="rk4", xp=backend.xp)

    y_gpu = to_cpu(y_gpu)

    match = np.allclose(y_cpu, y_gpu, rtol=1e-10, atol=1e-12)
    print("Match:", match)                                                                                                                                                                                                                          
                            
# Convergence analysis functions
                                                                                                                                                                                                                                     
def run_convergence():
    print("\n=== Convergence Study ===")

    methods = ["euler", "rk2", "rk4"]
    dts = [2**(-n) for n in range(4, 11)]

    results = []

    for method in methods:
        for dt in dts:

            y = solve_ivp(exp_decay, 1.0, dt, method=method, xp=np)
            err = abs(y - exp_decay_solution(1))

            results.append([method, dt, err])

    df = pd.DataFrame(results, columns=["method", "dt", "error"])
    df.to_csv("problem_1/prob1_errors.csv", index=False)

    print("Saved errors.csv")
    print(df)

    return df


# Fitting convergence orders

def fit_order(dts, errors):
    x = np.log(dts)
    y = np.log(errors)
    slope, _ = np.polyfit(x, y, 1)
    return slope


def compute_orders(df):
    print("\n=== Convergence Orders ===")

    methods = df.method.unique()
    table = []

    for m in methods:
        sub = df[df.method == m]
        order = fit_order(sub.dt.values, sub.error.values)
        table.append([m, order, "expected: 1,2,4"])

    df_orders = pd.DataFrame(
        table, columns=["Method", "Convergence Order", "Notes"]
    )

    print(df_orders)
    df_orders.to_csv("problem_1/prob1_orders.csv", index=False)

    return df_orders


# Timing study

def run_timing():
    print("\n=== Timing Study ===")

    methods = ["euler", "rk2", "rk4"]
    dts = [2**(-n) for n in range(4, 11)]

    backend = get_backend(prefer_gpu=True)
    
    # warm-up GPU to avoid startup overhead in timing
    solve_ivp(exp_decay, 1.0, 0.1, method="rk4", xp=backend.xp)
    backend.xp.cuda.Stream.null.synchronize()
    
    rows = []

    for m in methods:
        for dt in dts:

            # CPU timing
            with Timer(get_backend(False)) as t:
                solve_ivp(exp_decay, 1.0, dt, method=m, xp=np)
            cpu_time = t.dt

            # GPU timing
            with Timer(backend) as t:
                solve_ivp(exp_decay, 1.0, dt, method=m, xp=backend.xp)
            gpu_time = t.dt

            speedup = cpu_time / gpu_time if gpu_time > 0 else np.nan
            rows.append([m, dt, cpu_time, gpu_time, speedup])

    df = pd.DataFrame(rows, columns=["method","dt","cpu","gpu","speedup"])
    df.to_csv("problem_1/prob1_timings.csv", index=False)

    print("Saved timings.csv")
    print(df)

    return df


# Plotting functions
def plot_convergence(df):

    plt.figure()

    for m in df.method.unique():
        sub = df[df.method == m]
        plt.loglog(sub.dt, sub.error, marker='o', label=m)

    ref = np.array(sorted(df.dt.unique()))

    plt.loglog(ref, ref, '--', label='O(dt)')
    plt.loglog(ref, ref**2, '--', label='O(dt^2)')
    plt.loglog(ref, ref**4, '--', label='O(dt^4)')

    plt.xlabel("dt")
    plt.ylabel("Error")
    plt.title("Convergence Study")
    plt.legend()

    plt.savefig("problem_1/prob1_convergence.png", dpi=200)
    plt.show()


def plot_timing(df):
    
    plt.figure()

    for m in df.method.unique():
        sub = df[df.method == m]

        plt.loglog(sub.dt, sub.cpu, marker='o', label=f"{m}-CPU")
        plt.loglog(sub.dt, sub.gpu, marker='x', label=f"{m}-GPU")

    plt.xlabel("dt")
    plt.ylabel("Time (s)")
    plt.title("Timing Study")
    plt.legend()

    plt.savefig("problem_1/prob1_timing.png", dpi=200)
    plt.show()


# Speedup Report
def report_speedup(df):
    print("\n=== Speedup (smallest dt) ===")

    dt_min = df.dt.min()

    for m in df.method.unique():
        sub = df[(df.method == m) & (df.dt == dt_min)]

        cpu = sub.cpu.values[0]
        gpu = sub.gpu.values[0]

        speedup = cpu / gpu if gpu > 0 else np.nan

        print(f"{m}: CPU={cpu:.4e}, GPU={gpu:.4e}, Speedup={speedup:.2f}")

                                                                                                                                                          
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cpu", "gpu", "both"], default="both")
    args = parser.parse_args()

    if args.mode in ["cpu", "both"]:
        print("\n===== CPU RUN =====")
        run_validation()
        df_err = run_convergence()
        plot_convergence(df_err)
        
        compute_orders(df_err)
        df_time = run_timing()
        plot_timing(df_time)

        report_speedup(df_time)

    if args.mode in ["gpu", "both"]:
        print("\n===== GPU RUN =====")
        # GPU-specific timing or validation if needed
        backend = get_backend(True)
        print("GPU available:", backend.has_gpu)
        

