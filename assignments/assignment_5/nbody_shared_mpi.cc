#include <iostream> // Standard I/O
#include <fstream>  // File I/O
#include <random>   // Random number generators
#include <vector>   // Vector (dynamic array)
#include <tuple>    // Tuple (multiple return values)
#include <chrono>   // Time utilities
#include <mpi.h>    // MPI

// Global constants
static int N = 512;          // Number of masses
static const int D = 3;      // Dimensionality
static int ND = N * D;       // Size of the state vectors
static const double G = 0.5; // Gravitational constant
static const double dt = 1e-3; // Time step size
static const int T = 300;    // Number of time steps
static const double t_max = static_cast<double>(T) * dt; // Maximum time
static const double x_min = 0.; // Minimum position
static const double x_max = 1.; // Maximum position
static const double v_min = 0.; // Minimum velocity
static const double v_max = 0.; // Maximum velocity
static const double m_0 = 1.;   // Mass value
static const double epsilon = 0.01; // Softening parameter
static const double epsilon2 = epsilon * epsilon; // Softening parameter^2

using Vec = std::vector<double>; // Vector type

// Random number generator
static std::mt19937 gen; // Mersenne twister engine

// MPI variables
static int rank, n_ranks;        // Process rank and number of ranks
static int N_beg, N_end, N_local;    // Mass range per process
static int ND_beg, ND_end, ND_local;  // State vector range per process

// Shared memory pointers
static double *m = nullptr;
static double *x = nullptr;
static double *v = nullptr;
static double *a = nullptr;
static double *x_next = nullptr;
static double *v_next = nullptr;

// Shared windows
static MPI_Win win_m, win_x, win_v, win_a, win_x_next, win_v_next;

// Print a vector to a file
template <typename T>
void save(const std::vector<T>& vec, const std::string& filename,
          const std::string& header = "") {
    std::ofstream file(filename);
    if (file.is_open()) {
        if (!header.empty()) file << "# " << header << std::endl;
        for (const auto& elem : vec) file << elem << " ";
        file << std::endl;
        file.close();
    } else {
        std::cerr << "Unable to open file " << filename << std::endl;
    }
}

// Set up parallelism and shared memory
void setup_parallelism(int N_total) {
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    // Divide masses among ranks
    N_local = N_total / n_ranks + (rank < N_total % n_ranks ? 1 : 0);
    N_beg = rank * (N_total / n_ranks) + std::min(rank, N_total % n_ranks);
    N_end = N_beg + N_local;
    ND_local = N_local * D;
    ND_beg = N_beg * D;
    ND_end = ND_beg + ND_local;

    // Allocate shared memory
    if (rank == 0) {
        MPI_Win_allocate_shared(N_total * sizeof(double), sizeof(double),
                                MPI_INFO_NULL, MPI_COMM_WORLD, &m, &win_m);
        MPI_Win_allocate_shared(N_total * D * sizeof(double), sizeof(double),
                                MPI_INFO_NULL, MPI_COMM_WORLD, &x, &win_x);
        MPI_Win_allocate_shared(N_total * D * sizeof(double), sizeof(double),
                                MPI_INFO_NULL, MPI_COMM_WORLD, &v, &win_v);
        MPI_Win_allocate_shared(N_total * D * sizeof(double), sizeof(double),
                                MPI_INFO_NULL, MPI_COMM_WORLD, &a, &win_a);
        MPI_Win_allocate_shared(N_total * D * sizeof(double), sizeof(double),
                                MPI_INFO_NULL, MPI_COMM_WORLD, &x_next, &win_x_next);
        MPI_Win_allocate_shared(N_total * D * sizeof(double), sizeof(double),
                                MPI_INFO_NULL, MPI_COMM_WORLD, &v_next, &win_v_next);
    } else {
        int disp_unit;
        MPI_Aint size;
        MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL,
                                MPI_COMM_WORLD, &m, &win_m);
        MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL,
                                MPI_COMM_WORLD, &x, &win_x);
        MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL,
                                MPI_COMM_WORLD, &v, &win_v);
        MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL,
                                MPI_COMM_WORLD, &a, &win_a);
        MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL,
                                MPI_COMM_WORLD, &x_next, &win_x_next);
        MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL,
                                MPI_COMM_WORLD, &v_next, &win_v_next);
    }

    // Synchronize
    MPI_Barrier(MPI_COMM_WORLD);

    // Seed RNG
    auto now = std::chrono::high_resolution_clock::now();
    auto now_cast = std::chrono::time_point_cast<std::chrono::microseconds>(now);
    gen.seed(now_cast.time_since_epoch().count() ^ rank);
}

// Initialize positions and velocities
void initial_conditions() {
    std::mt19937 local_gen(1234 + rank);
    std::uniform_real_distribution<> ran_pos(x_min, x_max);
    std::uniform_real_distribution<> ran_vel(v_min, v_max);

    for (int i = ND_beg; i < ND_end; ++i) {
        x[i] = ran_pos(local_gen);
        v[i] = ran_vel(local_gen);
        m[i / D] = m_0;
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

// Compute accelerations
void compute_acceleration() {
    for (int i = ND_beg; i < ND_end / D * D; i += D) {
        int idx = i;
        for (int k = 0; k < D; ++k) a[idx + k] = 0.0;

        for (int j = 0; j < N; ++j) {
            double dx[D];
            double dx2 = epsilon2;
            for (int k = 0; k < D; ++k) {
                dx[k] = x[j*D + k] - x[idx + k];
                dx2 += dx[k]*dx[k];
            }
            double factor = G * m[j] / (dx2 * sqrt(dx2));
            for (int k = 0; k < D; ++k)
                a[idx + k] += factor * dx[k];
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// Perform a single timestep
void timestep() {
    compute_acceleration();

    for (int i = ND_beg; i < ND_end; ++i) {
        v_next[i] = v[i] + a[i] * dt;
        x_next[i] = x[i] + v_next[i] * dt;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Swap pointers
    std::swap(x, x_next);
    std::swap(v, v_next);

    MPI_Barrier(MPI_COMM_WORLD);
}

// Compute kinetic energy for local ranks
double kinetic_energy_local() {
    double KE_local = 0.0;
    for (int i = N_beg; i < N_end; ++i) {
        double v2 = 0.0;
        for (int k = 0; k < D; ++k)
            v2 += v[i*D + k] * v[i*D + k];
        KE_local += 0.5 * m[i] * v2;
    }
    return KE_local;
}

// Free shared memory
void free_shared_memory() {
    MPI_Win_free(&win_v_next);
    MPI_Win_free(&win_x_next);
    MPI_Win_free(&win_a);
    MPI_Win_free(&win_v);
    MPI_Win_free(&win_x);
    MPI_Win_free(&win_m);
}

int main(int argc, char** argv) {
    if (argc > 1) N = std::atoi(argv[1]);
    ND = N * D;

    setup_parallelism(N);
    initial_conditions();

    std::vector<double> t(T+1), KE(T+1);
    for (int i = 0; i <= T; ++i) t[i] = i*dt;

    for (int n = 0; n < T; ++n) {
        timestep();
        double KE_local = kinetic_energy_local();
        MPI_Reduce(&KE_local, &KE[n+1], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        save(KE, "outputs/KE_" + std::to_string(N) + ".txt", "Kinetic Energy");
        save(t, "outputs/time_" + std::to_string(N) + ".txt", "Time");
    }

    free_shared_memory();
    MPI_Finalize();
    return 0;
}
