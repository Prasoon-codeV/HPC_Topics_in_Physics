from numpy.random import rand
import sys

def calc_pi_loop(n):
    h = 0 # Number of hits inside the circle

    for _ in range(n):
        x, y = rand(), rand() # Random points in [0, 1)
    
        if x*x + y*y < 1.:
            h += 1 # Successful hit
    
    return 4. * float(h) / float(n) # Estimate pi

if __name__ == "__main__":
    num = sys.argv[1]
    
    n = int(float(num)) # Command-line argument
    
    if len(sys.argv) == 1:
        n = 13
    elif len(sys.argv) == 2:
        n = int(float(sys.argv[1])) # will allow inputs as exponents, 1e6 or so

    else:
        print('More arguments given than needed. Input arguments: <n>')

    pi_est = calc_pi_loop(n)
        
    with open(f'pi_numpy_{num}.txt', 'w') as f:
        f.write('\nValue of N, and pi_est:\n')
        f.write(f'{n:e}    {pi_est}')
        f.write('\n')
        
    print()
    print(f'Value of N={n:0.0e} and pi={pi_est:0.4e}')
        
    
