import numpy as np
import matplotlib.pyplot as plt

# Load HPC results
N = [10, 20, 40, 80] # Adjust as needed (use the largest N you ran on HPC)
potential = 'harmonic' # 'harmonic'

E = []

plt.figure(figsize=(6,4))
for n in N:
    data = np.loadtxt(f'outputs/eigs_N{n}_{potential}.txt')    
    E.append(data)    
    plt.plot(data, 'o-', label=f'N={n}')

plt.legend()
plt.xlabel('Eigenvalue Index')
plt.ylabel('Energy (arbitrary units)')
plt.title(f'Eigenvalues from HPC run')
plt.grid()
plt.show()
plt.savefig(f'outputs/{potential}_eigenvalues.pdf', dpi=300)


# Harmonic Potential

x = np.linspace(-1, 1, 80)
y = np.linspace(-1, 1, 80)
X, Y = np.meshgrid(x, y)
V = 4. * (X**2 + Y**2)
plt.figure()
plt.imshow(V.T, origin='lower', extent=(-1, 1, -1, 1))
plt.colorbar()
plt.title(f'Harmonic Potential V(x,y)')
plt.show()
plt.savefig(f'outputs/{potential}_potential.pdf', dpi=300)


# Eigenvalue Convergence

E0 = [i[0] for i in E]
E1 = [i[1] for i in E]
E2 = [i[2] for i in E]
E3 = [i[3] for i in E]
E4 = [i[4] for i in E]

## Normal plot
plt.figure(figsize=(6,4))
plt.plot(N, E0, 'o-', label='ground')
plt.plot(N, E1, 'o-', label='1st')
plt.plot(N, E2, 'o-', label='2nd')
plt.plot(N, E3, 'o-', label='3rd')
plt.plot(N, E4, 'o-', label='4th')

plt.xlabel('Grid size N')
plt.ylabel('Energy')
plt.title('Eigenvalue convergence')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig(f'outputs/{potential}_convergence.pdf', dpi=300)


## Log-log plot

plt.figure(figsize=(6,4))
plt.loglog(N, np.abs(np.array(E0)-E0[-1]), 'o-')
plt.loglog(N, np.abs(np.array(E1)-E1[-1]), 'o-')
plt.loglog(N, np.abs(np.array(E2)-E2[-1]), 'o-')
plt.loglog(N, np.abs(np.array(E3)-E3[-1]), 'o-')
plt.loglog(N, np.abs(np.array(E4)-E4[-1]), 'o-')
plt.xlabel('N')
plt.ylabel('|E(N)-E_ref|')
plt.title('Ground state convergence')
plt.grid()
plt.show()
plt.savefig(f'outputs/{potential}_convergence_loglog.pdf', dpi=300)

# N2 -> N,N
n = 40
x = np.linspace(-1, 1, n)
y = np.linspace(-1, 1, n)

X, Y = np.meshgrid(x, y)
V = 4. * (X**2 + Y**2)

plt.figure()
plt.imshow(V.T, origin='lower', extent=(-1, 1, -1, 1))
plt.colorbar()
plt.title('Harmonic Potential V(x,y)')
plt.show()
plt.savefig(f"outputs/N_N_harmonic_potential.png")

