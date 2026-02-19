import numpy as np
from src.eigen import solve_eigen

def test_small_grid():
    vals, _ = solve_eigen(N=5, potential='well', n_eigs=3)
    assert len(vals) == 3
    
    # Basic check: eigenvalues should be ascending
    assert np.all(np.diff(vals) >= 0), "Eigenvalues are not sorted"
    print('--- Run Successful for test/tests.py---')


def test_n_eigs():
    vals, _ = solve_eigen(N=2, potential='well', n_eigs=5)
    
    # Basic check: number of eigenvalues shouldn't exceed N**N
    assert len(vals) == 4, "Upper limit of size of Eigenvalues"
    print('--- Run Successful for test/tests.py---')

