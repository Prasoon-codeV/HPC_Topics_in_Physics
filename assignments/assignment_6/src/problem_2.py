import numpy as np
import matplotlib.pyplot as plt



# ---------------------  Stability functions R(z) -----------------------

def R_euler(z):
    return 1 + z

def R_rk2(z):
    return 1 + z + (z**2)/2

def R_rk4(z):
    return 1 + z + (z**2)/2 + (z**3)/6 + (z**4)/24


# --------------------- Compute stability region -----------------------

def compute_stability_grid(N=400, xlim=(-5,5), ylim=(-5,5)):

    x = np.linspace(xlim[0], xlim[1], N)
    y = np.linspace(ylim[0], ylim[1], N)

    X, Y = np.meshgrid(x, y)
    Z = X + 1j*Y

    return X, Y, Z


# ---------------------- Plot stability regions ------------------------

def plot_stability():

    X, Y, Z = compute_stability_grid()

    R1 = np.abs(R_euler(Z))
    R2 = np.abs(R_rk2(Z))
    R4 = np.abs(R_rk4(Z))

    plt.figure(figsize=(8,6))

    # Contour |R(z)| = 1
    plt.contour(X, Y, R1, levels=[1], linewidths=2, label='Euler', color='red')
    plt.contour(X, Y, R2, levels=[1], linewidths=2, linestyles='dashed', label='RK2', color='blue')
    plt.contour(X, Y, R4, levels=[1], linewidths=2, linestyles='dashdot', label='RK4', color='black')

    plt.axhline(0)
    plt.axvline(0)

    plt.title("Stability Regions |R(z)| = 1")
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")

    # Manual legend (matplotlib contour doesn't auto-label well)
    from matplotlib.lines import Line2D
    legend_lines = [
        Line2D([0], [0], color='blue', lw=2),
        Line2D([0], [0], color='blue', lw=2, linestyle='dashed'),
        Line2D([0], [0], color='blue', lw=2, linestyle='dashdot')
    ]

    plt.legend(legend_lines, ['Euler', 'RK2', 'RK4'])

    plt.grid()
    plt.savefig("problem_2/prob2_stability_regions.png", dpi=200)
    plt.show()


# ----------- Real-axis stability limit ------------------------

def estimate_real_axis_limit(R_func, name):

    x_vals = np.linspace(-5, 0, 10000)
    vals = np.abs(R_func(x_vals))

    stable = vals <= 1

    if np.any(stable):
        xmin = x_vals[stable][0]
    else:
        xmin = None

    print(f"{name}: xmin ~=  {xmin:.4f}")

    return xmin


def compute_real_axis_limits():

    print("\n=== Real-axis stability limits ===")

    euler = estimate_real_axis_limit(R_euler, "Euler")
    rk2   = estimate_real_axis_limit(R_rk2, "RK2")
    rk4   = estimate_real_axis_limit(R_rk4, "RK4")

    return euler, rk2, rk4


# ----------------- Problem 2 ------------------------------

if __name__ == "__main__":

    plot_stability()
    compute_real_axis_limits()




