#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <vector>
#include <functional>
#include <mpi.h>

double Lx = 1.0;
double Ly = 1.0;
double Lz = 1.0;

int Nx = 128, Ny = 128, Nz = 128;
double T = 0.001;
int Nt = 20;
double tau = T / Nt;

double hx = Lx / (Nx - 1), hy = Ly / (Ny - 1), hz = Lz / (Nz - 1);

double at = M_PI * std::sqrt(9.0 / (Lx * Lx) + 4.0 / (Ly * Ly) + 4.0 / (Lz * Lz));

std::vector<double> xCoordinates, yCoordinates, zCoordinates;

int localX, localY, localZ;

std::vector<double> sendLeftX, recvLeftX,
                    sendRightX, recvRightX,
                    sendLeftY, recvLeftY,
                    sendRightY, recvRightY,
                    sendLeftZ, recvLeftZ,
                    sendRightZ, recvRightZ;

double analyticalSolution(double x, double y, double z, double t) {
    return sin(3 * M_PI * x / Lx) *
           sin(2 * M_PI * y / Ly) *
           sin(2 * M_PI * z / Lz) *
           cos(at * t + 4 * M_PI);
}

int getIndex(int i, int j, int k) {
    return i * localY * localZ + j * localZ + k;
}

void initCoordinates(int dimensions[3], int coordinates[3]) {
    int startIndex[3], endIndex[3], localSizes[3];
    int sizes[3] = {Nx, Ny, Nz};

    for (int i = 0; i < 3; ++i) {
        localSizes[i] = sizes[i] / dimensions[i];
        startIndex[i] = localSizes[i] * coordinates[i];
        endIndex[i] = localSizes[i] * (coordinates[i] + 1);
        if (coordinates[i] == dimensions[i] - 1)
            endIndex[i] = sizes[i];
    }

    for (int i = startIndex[0]; i < endIndex[0]; i++)
        xCoordinates.push_back(i * hx);
    for (int i = startIndex[1]; i < endIndex[1]; i++)
        yCoordinates.push_back(i * hy);
    for (int i = startIndex[2]; i < endIndex[2]; i++)
        zCoordinates.push_back(i * hz);
}

inline double computeSecondDerivative(const std::vector<double>& u,
                         const std::vector<double>& recvLeft,
                         const std::vector<double>& recvRight,
                         int index, int upperBoundary,
                         double hSquared,
                         int centerIndex,
                         int leftIndex,
                         int rightIndex,
                         int recvIndex) {
    if (index == 0) 
        return (recvLeft[recvIndex] - 2.0 * u[centerIndex] + u[rightIndex]) / hSquared;

    if (index == upperBoundary)
        return (u[leftIndex] - 2.0 * u[centerIndex] + recvRight[recvIndex]) / hSquared;

    return (u[leftIndex] - 2.0 * u[centerIndex] + u[rightIndex]) / hSquared;
    
}

double computeLaplacian(const std::vector<double>& u, int i, int j, int k) {
    int idx = getIndex(i, j, k);
    double derivativeX, derivativeY, derivativeZ;

    derivativeX = computeSecondDerivative(u, recvLeftX, recvRightX, 
                     i, localX - 1, hx * hx, 
                     idx, getIndex(i - 1, j, k), getIndex(i + 1, j, k), 
                     j * localZ + k);

    derivativeY = computeSecondDerivative(u, recvLeftY, recvRightY, 
                     j, localY - 1, hy * hy, 
                     idx, getIndex(i, j - 1, k), getIndex(i, j + 1, k), 
                     i * localZ + k);

    derivativeZ = computeSecondDerivative(u, recvLeftZ, recvRightZ, 
                     k, localZ - 1, hz * hz, 
                     idx, getIndex(i, j, k - 1), getIndex(i, j, k + 1), 
                     i * localY + j);

    return derivativeX + derivativeY + derivativeZ;
}

void updateField(
    std::vector<double>& targetU,
    std::function<double(int i, int j, int k)> updateFunction
) {
    for(int i = 0; i < localX; i++)
        for(int j = 0; j < localY; j++)
            for(int k = 0; k < localZ; k++) {
                double x = xCoordinates[i];

                if (x != 0 && x != Lx)
                    targetU[getIndex(i, j, k)] = updateFunction(i, j, k);
            }
}

double computeMaximumError(const std::vector<double>& targetU, double t) {
    double maxError = 0.0;

    for(int i = 0; i < localX; i++)
        for(int j = 0; j < localY; j++)
            for(int k = 0; k < localZ; k++) {
                double x = xCoordinates[i], y = yCoordinates[j], z = zCoordinates[k];
                int idx = getIndex(i, j, k);

                double error = std::abs(targetU[idx] - analyticalSolution(x, y, z, t));
                if (error > maxError)
                    maxError = error;
            }
    return maxError;
}

inline void prepareSendBuffer(const std::vector<double>& targetU, 
                                 std::vector<double>& sendLeft,
                                 std::vector<double>& sendRight,
                                 int dim1, int dim2, int dim3, 
                                 std::function<int(int index, int ind1, int ind2)> getIndexFunc) {

    for (int i = 0; i < dim1; i++)
        for (int j = 0; j < dim2; j++) {
            int idx = i * dim2 + j;

            sendLeft[idx] = targetU[getIndexFunc(0, i, j)];
            sendRight[idx] = targetU[getIndexFunc(dim3 - 1, i, j)];
        }
}

inline void prepareAllSendBuffers(const std::vector<double>& targetU) {
    prepareSendBuffer(targetU, sendLeftX, sendRightX, localY, localZ, localX,
                         [](int index, int j, int k) { return getIndex(index, j, k); });

    prepareSendBuffer(targetU, sendLeftY, sendRightY, localX, localZ, localY,
                         [](int index, int i, int k) { return getIndex(i, index, k); });

    prepareSendBuffer(targetU, sendLeftZ, sendRightZ, localX, localY, localZ,
                         [](int index, int i, int j) { return getIndex(i, j, index); });

}

inline void communicateLayer(MPI_Comm comm, MPI_Request requests[], 
                              int& requestCount, int count, int sourcePrev, int sourceNext, int tag,
                              std::vector<double>& sendLeft, std::vector<double>& recvLeft, 
                              std::vector<double>& sendRight, std::vector<double>& recvRight) {
    if (sourcePrev != MPI_PROC_NULL) {
        MPI_Irecv(recvLeft.data(), count, MPI_DOUBLE, sourcePrev, tag, comm, &requests[requestCount++]);
        MPI_Isend(sendLeft.data(), count, MPI_DOUBLE, sourcePrev, tag + 1, comm, &requests[requestCount++]);
    }
    if (sourceNext != MPI_PROC_NULL) {
        MPI_Irecv(recvRight.data(), count, MPI_DOUBLE, sourceNext, tag + 1, comm, &requests[requestCount++]);
        MPI_Isend(sendRight.data(), count, MPI_DOUBLE, sourceNext, tag, comm, &requests[requestCount++]);
    }
}

void communicateBoundaryLayers(const std::vector<double>& targetU,
                              MPI_Comm comm, const int rank_prev[3], const int rank_next[3]) {
    prepareAllSendBuffers(targetU);

    MPI_Request requests[12];
    int requestCount = 0;

    communicateLayer(comm, requests, requestCount, localY * localZ, rank_prev[0], rank_next[0], 0, 
                      sendLeftX, recvLeftX, sendRightX, recvRightX);
    communicateLayer(comm, requests, requestCount, localX * localZ, rank_prev[1], rank_next[1], 2, 
                      sendLeftY, recvLeftY, sendRightY, recvRightY);
    communicateLayer(comm, requests, requestCount, localX * localY, rank_prev[2], rank_next[2], 4, 
                      sendLeftZ, recvLeftZ, sendRightZ, recvRightZ);

    MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);
}

void reportError(MPI_Comm comm, double localMaxError, double& globalMaxError, int rank, int timestep, bool isFinal = false) {
    MPI_Reduce(&localMaxError, &globalMaxError, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        if (!isFinal)
            std::cout << "\nTime step: " << timestep << " " << "Max err: " << globalMaxError;
        else
            std::cout << "\n\nTotal time: " << globalMaxError << " seconds"; 
    }
}

int main(int argc, char** argv) {
    if (argc > 1) {
        try {
            double grid_length = std::stod(argv[1]);
            Nx = Ny = Nz = grid_length;
            hx = Lx / (Nx - 1), hy = Ly / (Ny - 1), hz = Lz / (Nz - 1);
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

        hx = Lx / (Nx - 1), hy = Ly / (Ny - 1), hz = Lz / (Nz - 1);

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

    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double startTime = MPI_Wtime();
    double globalMaxError;

    int dimentionsCount = 3;

    MPI_Comm comm;
    int gridDimensions[3] = {0, 0, 0};
    int periodicDirections[3] = {0, 1, 1};
    int gridCoordinates[3];

    MPI_Dims_create(size, dimentionsCount, gridDimensions);
    MPI_Cart_create(MPI_COMM_WORLD, dimentionsCount, gridDimensions, periodicDirections, 0, &comm);
    MPI_Cart_coords(comm, rank, dimentionsCount, gridCoordinates);

    int neighborRanksPrev[dimentionsCount], neighborRanksNext[dimentionsCount];

    for (int i = 0; i < dimentionsCount; ++i) {
        MPI_Cart_shift(comm, i, 1, &neighborRanksPrev[i], &neighborRanksNext[i]);
    }

    initCoordinates(gridDimensions, gridCoordinates);

    localX = xCoordinates.size();
    localY = yCoordinates.size();
    localZ = zCoordinates.size();

    int totalLocalPoints = localX * localY * localZ;

    std::vector<double> u0(totalLocalPoints, 0.0);
    std::vector<double> u1(totalLocalPoints, 0.0);
    std::vector<double> u2(totalLocalPoints, 0.0);

    updateField(
            u0,
            [](int i, int j, int k) {
                return analyticalSolution(xCoordinates[i], yCoordinates[j], zCoordinates[k], 0.0);
            }
        );

    int layerSizeXY = localX * localY, layerSizeXZ = localX * localZ, layerSizeYZ = localY * localZ;

    sendLeftX.resize(layerSizeYZ); recvLeftX.resize(layerSizeYZ);
    sendRightX.resize(layerSizeYZ); recvRightX.resize(layerSizeYZ);

    sendLeftY.resize(layerSizeXZ); recvLeftY.resize(layerSizeXZ); 
    sendRightY.resize(layerSizeXZ); recvRightY.resize(layerSizeXZ);

    sendLeftZ.resize(layerSizeXY); recvLeftZ.resize(layerSizeXY);
    sendRightZ.resize(layerSizeXY); recvRightZ.resize(layerSizeXY);

    communicateBoundaryLayers(u0, comm, neighborRanksPrev, neighborRanksNext);
    
    
    updateField(
            u1,
            [&u0]
            (int i, int j, int k) {
                return u0[getIndex(i, j, k)] + tau * tau * 0.5 * computeLaplacian(u0, i, j, k);
            }
        );

    reportError(comm, computeMaximumError(u1, tau), globalMaxError, rank, 0);

    for (int timestep = 1; timestep < Nt; timestep++) {
        communicateBoundaryLayers(u1, comm, neighborRanksPrev, neighborRanksNext);
        updateField(
            u2,
            [&u1, &u0]
            (int i, int j, int k) {
                int idx = getIndex(i, j, k);

                return 2 * u1[idx] - u0[idx] + tau * tau * computeLaplacian(u0, i, j, k);
            }
        );

        u0.swap(u1);
        u1.swap(u2);

        reportError(comm, computeMaximumError(u1, tau * (timestep + 1)), globalMaxError, rank, timestep + 1);
    }

    double endTime = MPI_Wtime();
    double elapsedTime = endTime - startTime, maxElapsedTime;
    
    reportError(comm, elapsedTime, maxElapsedTime, rank, Nt, true);

    MPI_Finalize();

    return 0;
}
