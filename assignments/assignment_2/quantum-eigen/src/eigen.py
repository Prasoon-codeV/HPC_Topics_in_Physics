import numpy as np
from scipy.linalg import eigh
import argparse
import matplotlib.pyplot as plt

VALID_POTENTIALS = ['well', 'harmonic', 'shell_oscillator']
def build_2d_hamiltonian(N=20, potential='well'):
    """
    Build a discretized 2D Hamiltonian on an N x N grid.
    Parameters
    ----------
    N : int
    potential : str
    Number of points in each dimension (N^2 total points).
    Choose the potential. 'well' or 'harmonic' or 'shell_oscillator' examples.
    Returns
    -------
    H : ndarray of shape (N^2, N^2)
    The Hamiltonian matrix approximating -d^2/dx^2 - d^2/dy^2 + V(x,y).
    """
    dx = 1. / float(N) # grid spacing, can be arbitrary
    inv_dx2 = float(N * N) # 1/dx^2
    H = np.zeros((N*N, N*N), dtype=np.float64)
    # Helper function to map (i,j) -> linear index
    def idx(i, j):
        return i * N + j
        # Potential function
    def V(i, j):
        # Example 1: infinite square well -> zero in interior, large outside
        if potential == 'well':
            # No boundary enforcement here, but can skip boundary wavefunction
            return 0.
        # Example 2: 2D harmonic oscillator around center
        elif potential == 'harmonic':
            x = (i - N/2) * dx
            y = (j - N/2) * dx
            # Quadratic potential V = k * (x^2 + y^2)
            return 4. * (x**2 + y**2)
        elif potential == 'shell_oscillator':
            # Oscillator with shell: V=90V between R1 and R2, harmonic everywhere else
            x = (i - N/2) * dx
            y = (j - N/2) * dx

            # Cylindrical shell: V=90V between R1 and R2, harmonic everywhere else
            r = np.sqrt(x**2 + y**2)
            R1 = 0.15  # radius of inner boundary
            R2 = 0.25 # radius of outer boundary
            V = 4. * (x**2 + y**2)
            if (r < R1) or (r > R2):
                return V
            else:
                V = 90. # add barrier height in the shell region
                return V # large potential between the boundaries
        
        
        else:
            return 0.
    # Build the matrix: For each (i, j), set diagonal for 2D Laplacian plus V
    for i in range(N):
        for j in range(N):
            row = idx(i,j)
            # Potential
            H[row, row] = 4. * inv_dx2 + V(i,j) # "Kinetic" ~ -4/dx^2 in 2D FD
            # Neighbors (assuming no boundary conditions or Dirichlet)
            if i > 0: # up
                H[row, idx(i-1, j)] = inv_dx2
            if i < N-1: # down
                H[row, idx(i+1, j)] = inv_dx2
            if j > 0: # left
                H[row, idx(i, j-1)] = inv_dx2
            if j < N-1: # right
                H[row, idx(i, j+1)] = inv_dx2
    return H

def solve_eigen(N=20, potential='well', n_eigs=None, grnd=False):   
    H = build_2d_hamiltonian(N, potential)

    # Solve entire spectrum (careful for large N)
    vals, vecs = eigh(H)
    # Sort
    idx_sorted = np.argsort(vals)
    vals_sorted = vals[idx_sorted]
    vecs_sorted = vecs[:, idx_sorted]
    
    if grnd:
        if n_eigs is None:
            return vals_sorted, vecs_sorted, vecs_sorted[:, 0]
        else:
            return vals_sorted[:n_eigs], vecs_sorted[:, :n_eigs], vecs_sorted[:, 0]
    else:   
        if n_eigs is None:
            return vals_sorted, vecs_sorted
        else:
            return vals_sorted[:n_eigs], vecs_sorted[:, :n_eigs]

def plot_potential(n, potential): 

    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, y)
    V = np.zeros_like(X)
    
    if potential == 'harmonic':
        V = 4. * (X**2 + Y**2)
    elif potential == 'shell_oscillator':
        R1 = 0.25  # radius of inner boundary
        R2 = 0.35 # radius of outer boundary
        r = np.sqrt(X**2 + Y**2)
        V = 4. * (X**2 + Y**2)
        V[(r >= R1) & (r <= R2)] = 90.        

    plt.figure()
    plt.imshow(V.T, origin='lower', extent=(-1, 1, -1, 1))
    plt.colorbar()
    plt.title(f'{potential} Potential V(x,y)')
    plt.show()
    plt.savefig(f'{potential}_potential.pdf', dpi=300)
    plt.close()


if __name__ == '__main__':
    # Example local test

    parser = argparse.ArgumentParser(description="Solve 2D Hamiltonian eigenvalue problem.")
    
    parser.add_argument('--N', type=int, required=True)
    parser.add_argument('--potential', choices=VALID_POTENTIALS, required=True)
    parser.add_argument('--n-eigs', type=int, required=True)
    parser.add_argument('--grnd', type=bool, default=False)
    args = parser.parse_args()
    
    potential = args.potential
    N = args.N
    grnd = args.grnd
    n_eigs = args.n_eigs
    
    if n_eigs > args.N * args.N:
        print(f"Warning: Requested n_eigs={n_eigs} exceeds total number of states N^2={args.N * args.N}. Setting n_eigs to {args.N * args.N}.")
        n_eigs = args.N * args.N

    if args.grnd:
        vals, vecs, grnd_vec = solve_eigen(N=args.N, potential=args.potential, n_eigs=n_eigs, grnd=args.grnd)
    else:
        vals, vecs = solve_eigen(N=args.N, potential=args.potential, n_eigs=n_eigs)

    plot_potential(args.N, args.potential)

    print(f"Lowest {n_eigs} eigenvalues:", vals)
    
    np.savetxt(f'eigs_N{args.N}_{potential}.txt', vals)
    if args.grnd:
        psi = grnd_vec.reshape(args.N, args.N)
        psi_2d = psi**2
        print("Ground state wavefunction:", psi_2d)
        np.savetxt(f'grnd_psi_2d_N{args.N}.txt', psi_2d)
    
