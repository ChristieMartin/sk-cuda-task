#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <functional>
#include <omp.h>
#include <string>

typedef std::vector<std::vector<std::vector<double>>> Vector3D;

double Lx = M_PI, Ly = M_PI, Lz = M_PI;

int Nx = 128, Ny = 128, Nz = 128;

int Nt = 20;

double T = 0.001;
double tau = T / Nt;

double hx = Lx / (Nx - 1);
double hy = Ly / (Ny - 1);
double hz = Lz / (Nz - 1);

double at = M_PI * std::sqrt((9.0 / (Lx * Lx)) + (4.0 / (Ly * Ly)) + (4.0 / (Lz * Lz)));

double u_analytical(double x, double y, double z, double t) {
    return std::sin( (3.0 * M_PI * x) / Lx ) *
           std::sin( (2.0 * M_PI * y) / Ly ) *
           std::sin( (2.0 * M_PI * z) / Lz ) *
           std::cos( at * t + 4.0 * M_PI );
}

void init_zero(Vector3D& u) {
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                double x = i * hx;
                double y = j * hy;
                double z = k * hz;
                
                u[i][j][k] = u_analytical(x, y, z, 0);
            }
        }
    }
}

void compute_next_step(
    const Vector3D& u_source,
    Vector3D& u_target,
    std::function<double(int i, int j, int k, double laplacian)> update_formula
) {
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                if (i == 0 || i == Nx - 1) {
                    u_target[i][j][k] = 0.0;
                    continue;
                }

                double laplacian = (u_source[i + 1][j][k] - 2 * u_source[i][j][k] + u_source[i - 1][j][k]) / (hx * hx);
                
                int jp = (j + 1) % Ny;
                int jm = (j - 1 + Ny) % Ny;
                laplacian += (u_source[i][jp][k] - 2 * u_source[i][j][k] + u_source[i][jm][k]) / (hy * hy);

                int kp = (k + 1) % Nz;
                int km = (k - 1 + Nz) % Nz;
                laplacian += (u_source[i][j][kp] - 2 * u_source[i][j][k] + u_source[i][j][km]) / (hz * hz);

                u_target[i][j][k] = update_formula(i, j, k, laplacian);
            }
        }
    }
}

double compute_max_error(const Vector3D& u, double t) {
    double max_error = 0.0;
    #pragma omp parallel for collapse(3) reduction(max:max_error)
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                double x = i * hx;
                double y = j * hy;
                double z = k * hz;

                double u_an = u_analytical(x, y, z, t);
                double error = std::abs(u[i][j][k] - u_an);

                if (error > max_error) {
                    max_error = error;
                }
            }
        }
    }
    return max_error;
}

int main(int argc, char* argv[]) {
    if (argc > 1) {
        try {
            double grid_length = std::stod(argv[1]);
            Nx = Ny = Nz = grid_length;
        } catch (...) {
            std::cerr << "Provide a number as a first argument" << std::endl;
            return 1;
        }
 
    }

    if (argc > 2) {
        std::string arg = argv[2];
        if (arg == "pi") {
            Lx = Ly = Lz = M_PI;
        } else {
            try {
                double length = std::stod(arg);
                Lx = Ly = Lz = length;
            } catch (...) {
                std::cerr << "Provide a number or 'pi' as second argument." << std::endl;
                return 1;
            }
        }

        hx = Lx / (Nx - 1);
        hy = Ly / (Ny - 1);
        hz = Lz / (Nz - 1);

        at = M_PI * std::sqrt( (9.0 / (Lx * Lx)) + (4.0 / (Ly * Ly)) + (4.0 / (Lz * Lz)) );
    }

    if (argc > 3) {
        try {
            T = std::stod(argv[3]);
            tau = T / Nt;
        } catch (...) {
            std::cerr << "Provide a number as a third argument" << std::endl;
            return 1;
        }
    }

    Vector3D u0(Nx, std::vector<std::vector<double>>(Ny, std::vector<double>(Nz)));
    Vector3D u1(Nx, std::vector<std::vector<double>>(Ny, std::vector<double>(Nz)));
    Vector3D u2(Nx, std::vector<std::vector<double>>(Ny, std::vector<double>(Nz)));

    double max_error = 0.0;
    auto start_time = std::chrono::high_resolution_clock::now();

    init_zero(u0);

    compute_next_step(
        u0, u1,
        [&u0](int i, int j, int k, double laplacian) {
            return u0[i][j][k] + (tau * tau * 0.5) * laplacian;
        }
    );

    for (int n = 1; n < Nt; ++n) {
        compute_next_step(
            u1, u2,
            [&u1, &u0](int i, int j, int k, double laplacian) {
                return 2 * u1[i][j][k] - u0[i][j][k] + tau * tau * laplacian;
            }
        );

        std::swap(u0, u1);
        std::swap(u1, u2);

        double t = n * tau;

        max_error = compute_max_error(u1, t);

        std::cout << "Time step " << n << ", Max error: " << max_error << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;

    std::cout << "Elapsed time: " << elapsed_seconds.count() << " seconds, Max error: " << max_error << std::endl;

    return 0;
}
