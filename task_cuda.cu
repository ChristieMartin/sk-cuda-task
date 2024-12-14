#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <vector>
#include <functional>
#include <mpi.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
   if (code != cudaSuccess){
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__constant__ double d_Lx;
__constant__ double d_Ly;
__constant__ double d_Lz;
__constant__ double d_at;
__constant__ double d_hx;
__constant__ double d_hy;
__constant__ double d_hz;

const int numThreads = 128;

__host__ __device__
double analyticalSolution(double x, double y, double z, double t, double Lx, double Ly, double Lz, double at_val) {
    return sin(3 * M_PI * x / Lx) *
           sin(2 * M_PI * y / Ly) *
           sin(2 * M_PI * z / Lz) *
           cos(at_val * t + 4 * M_PI);
}

__host__ __device__
int getIndex(int i, int j, int k, int localY, int localZ) {
    return i * localY * localZ + j * localZ + k;
}

__global__
void initializeField(double* d_u, double* d_x, double* d_y, double* d_z, double Lx, int localX, int localY, int localZ, double t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < localX && j < localY && k < localZ) {
        int idx = getIndex(i, j, k, localY, localZ);
        double x = d_x[i];
        double y = d_y[j];
        double z = d_z[k];
        if (x != 0 && x != Lx) {
            d_u[idx] = analyticalSolution(x, y, z, t, d_Lx, d_Ly, d_Lz, d_at);
        } else {
            d_u[idx] = 0;
        }
    }
}

__global__
void computeLaplacianKernel(const double* d_u, double* d_laplacian, 
                            const double* recvLeftX, const double* recvRightX,
                            const double* recvLeftY, const double* recvRightY,
                            const double* recvLeftZ, const double* recvRightZ,
                            int localX, int localY, int localZ,
                            double hxsq, double hysq, double hzsq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < localX && j < localY && k < localZ) {
        int idx = getIndex(i, j, k, localY, localZ);
        double derivativeX, derivativeY, derivativeZ;

        if (i == 0)
            derivativeX = (recvLeftX[j * localZ + k] - 2.0 * d_u[idx] + d_u[getIndex(i + 1, j, k, localY, localZ)]) / hxsq;
        else if (i == localX - 1)
            derivativeX = (d_u[getIndex(i - 1, j, k, localY, localZ)] - 2.0 * d_u[idx] + recvRightX[j * localZ + k]) / hxsq;
        else
            derivativeX = (d_u[getIndex(i - 1, j, k, localY, localZ)] - 2.0 * d_u[idx] + d_u[getIndex(i + 1, j, k, localY, localZ)]) / hxsq;

        if (j == 0)
            derivativeY = (recvLeftY[i * localZ + k] - 2.0 * d_u[idx] + d_u[getIndex(i, j + 1, k, localY, localZ)]) / hysq;
        else if (j == localY - 1)
            derivativeY = (d_u[getIndex(i, j - 1, k, localY, localZ)] - 2.0 * d_u[idx] + recvRightY[i * localZ + k]) / hysq;
        else
            derivativeY = (d_u[getIndex(i, j - 1, k, localY, localZ)] - 2.0 * d_u[idx] + d_u[getIndex(i, j + 1, k, localY, localZ)]) / hysq;

        if (k == 0)
            derivativeZ = (recvLeftZ[i * localY + j] - 2.0 * d_u[idx] + d_u[getIndex(i, j, k + 1, localY, localZ)]) / hzsq;
        else if (k == localZ - 1)
            derivativeZ = (d_u[getIndex(i, j, k - 1, localY, localZ)] - 2.0 * d_u[idx] + recvRightZ[i * localY + j]) / hzsq;
        else
            derivativeZ = (d_u[getIndex(i, j, k - 1, localY, localZ)] - 2.0 * d_u[idx] + d_u[getIndex(i, j, k + 1, localY, localZ)]) / hzsq;

        d_laplacian[idx] = derivativeX + derivativeY + derivativeZ;
    }
}

__global__
void updateFieldKernel(
    double* d_targetU,
    const double* d_currentU,
    const double* d_previousU,
    const double* d_laplacian, 
    const double* d_x,
    double Lx, 
    double tau_sq, 
    int localX, 
    int localY, 
    int localZ
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < localX && j < localY && k < localZ) {
        double x = d_x[i];
        int idx = getIndex(i, j, k, localY, localZ);

        if (x != 0.0 && x != Lx) {
            d_targetU[idx] = 2.0 * d_currentU[idx] - d_previousU[idx] + tau_sq * d_laplacian[idx];
        } else {
            d_targetU[idx] = 0;
        }
    }
}

__global__
void initializeU1Kernel(
    double* d_targetU,
    const double* d_currentU,
    const double* d_laplacian, 
    const double* d_x,
    double Lx, 
    double tau_sq_half, 
    int localX, 
    int localY, 
    int localZ
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < localX && j < localY && k < localZ) {
        double x = d_x[i];
        int idx = getIndex(i, j, k, localY, localZ);

        if (x != 0.0 && x != Lx) {
            d_targetU[idx] = d_currentU[idx] + tau_sq_half * d_laplacian[idx];
        } else {
            d_targetU[idx] = 0;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    {

        int Nx = 128, Ny = 128, Nz = 128;
        double T = 0.001;
        int Nt = 20;
        double tau = T / Nt;

        double Lx = 1.0, Ly = 1.0, Lz = 1.0;
        double hx = 0.0, hy = 0.0, hz = 0.0;
        double at = 0.0;

        if (argc > 1) {
            try {
                double grid_length = std::stod(argv[1]);
                Nx = Ny = Nz = static_cast<int>(grid_length);
                if(rank == 0) std::cout << "Grid size set to " << Nx << " in each dimension." << std::endl;
            } catch (...) {
                if (rank == 0)
                    std::cerr << "Provide a number as a first argument" << std::endl;
                MPI_Finalize();
                return 1;
            }
        }

        if (argc > 2) {
            std::string arg = argv[2];
            if (arg == "pi") {
                Lx = Ly = Lz = M_PI;
                if(rank == 0) std::cout << "Domain length set to pi in each dimension." << std::endl;
            } else {
                try {
                    double length = std::stod(arg);
                    Lx = Ly = Lz = length;
                    if(rank == 0) std::cout << "Domain length set to " << length << " in each dimension." << std::endl;
                } catch (...) {
                    if (rank == 0)
                        std::cerr << "Provide a number or 'pi' as second argument." << std::endl;
                    MPI_Finalize();
                    return 1;
                }
            }
        }

        if (argc > 3) {
            try {
                T = std::stod(argv[3]);
                tau = T / Nt;
                if(rank == 0) std::cout << "Total time set to " << T << " with " << Nt << " time steps." << std::endl;
            } catch (...) {
                if (rank == 0)
                    std::cerr << "Provide a number as a third argument" << std::endl;
                MPI_Finalize();
                return 1;
            }
        }

        hx = Lx / (Nx - 1);
        hy = Ly / (Ny - 1);
        hz = Lz / (Nz - 1);

        at = M_PI * std::sqrt(9.0 / (Lx * Lx) + 4.0 / (Ly * Ly) + 4.0 / (Lz * Lz));


        cudaCheckError(cudaMemcpyToSymbol(d_Lx, &Lx, sizeof(double)));
        cudaCheckError(cudaMemcpyToSymbol(d_Ly, &Ly, sizeof(double)));
        cudaCheckError(cudaMemcpyToSymbol(d_Lz, &Lz, sizeof(double)));
        cudaCheckError(cudaMemcpyToSymbol(d_at, &at, sizeof(double)));
        cudaCheckError(cudaMemcpyToSymbol(d_hx, &hx, sizeof(double)));
        cudaCheckError(cudaMemcpyToSymbol(d_hy, &hy, sizeof(double)));
        cudaCheckError(cudaMemcpyToSymbol(d_hz, &hz, sizeof(double)));

        int dimensionsCount = 3;
        int gridDimensions[3] = {0, 0, 0};
        int periodicDirections[3] = {0, 1, 1};
        int gridCoordinates[3];

        MPI_Dims_create(size, dimensionsCount, gridDimensions);
        MPI_Comm comm;
        MPI_Cart_create(MPI_COMM_WORLD, dimensionsCount, gridDimensions, periodicDirections, 0, &comm);
        MPI_Cart_coords(comm, rank, dimensionsCount, gridCoordinates);

 

        int neighborRanksPrev[3], neighborRanksNext[3];
        for (int i = 0; i < dimensionsCount; ++i) {
            MPI_Cart_shift(comm, i, 1, &neighborRanksPrev[i], &neighborRanksNext[i]);
        }

        int sizes[3] = {Nx, Ny, Nz};
        int localSizes[3];
        int startIndex[3], endIndex[3];

        for (int i = 0; i < 3; ++i) {
            localSizes[i] = sizes[i] / gridDimensions[i];
            startIndex[i] = localSizes[i] * gridCoordinates[i];
            endIndex[i] = localSizes[i] * (gridCoordinates[i] + 1);
            if (gridCoordinates[i] == gridDimensions[i] - 1)
                endIndex[i] = sizes[i];
        }

        int localX = endIndex[0] - startIndex[0];
        int localY = endIndex[1] - startIndex[1];
        int localZ = endIndex[2] - startIndex[2];

        thrust::host_vector<double> xCoordinates, yCoordinates, zCoordinates;
        xCoordinates.reserve(localX);
        yCoordinates.reserve(localY);
        zCoordinates.reserve(localZ);

        for (int i = startIndex[0]; i < endIndex[0]; ++i)
            xCoordinates.push_back(i * hx);
        for (int i = startIndex[1]; i < endIndex[1]; ++i)
            yCoordinates.push_back(i * hy);
        for (int i = startIndex[2]; i < endIndex[2]; ++i)
            zCoordinates.push_back(i * hz);


        thrust::device_vector<double> d_x(localX);
        thrust::device_vector<double> d_y(localY);
        thrust::device_vector<double> d_z(localZ);

        thrust::copy(xCoordinates.begin(), xCoordinates.end(), d_x.begin());
        thrust::copy(yCoordinates.begin(), yCoordinates.end(), d_y.begin());
        thrust::copy(zCoordinates.begin(), zCoordinates.end(), d_z.begin());

        int totalLocalPoints = localX * localY * localZ;
        thrust::host_vector<double> h_u0(totalLocalPoints, 0.0);
        thrust::host_vector<double> h_u1(totalLocalPoints, 0.0);
        thrust::host_vector<double> h_u2(totalLocalPoints, 0.0);

        int layerSizeYZ = localY * localZ;
        int layerSizeXZ = localX * localZ;
        int layerSizeXY = localY * localX;

        thrust::host_vector<double> sendLeftX, recvLeftX,
                                 sendRightX, recvRightX,
                                 sendLeftY, recvLeftY,
                                 sendRightY, recvRightY,
                                 sendLeftZ, recvLeftZ,
                                 sendRightZ, recvRightZ;

        thrust::device_vector<double> recvLeftX_dev, recvRightX_dev,
                                  recvLeftY_dev, recvRightY_dev,
                                  recvLeftZ_dev, recvRightZ_dev;

        sendLeftX.resize(layerSizeYZ, 0.0); recvLeftX.resize(layerSizeYZ, 0.0);
        sendRightX.resize(layerSizeYZ, 0.0); recvRightX.resize(layerSizeYZ, 0.0);

        sendLeftY.resize(layerSizeXZ, 0.0); recvLeftY.resize(layerSizeXZ, 0.0);
        sendRightY.resize(layerSizeXZ, 0.0); recvRightY.resize(layerSizeXZ, 0.0);

        sendLeftZ.resize(layerSizeXY, 0.0); recvLeftZ.resize(layerSizeXY, 0.0);
        sendRightZ.resize(layerSizeXY, 0.0); recvRightZ.resize(layerSizeXY, 0.0);

        thrust::device_vector<double> d_u0(totalLocalPoints, 0.0);
        thrust::device_vector<double> d_u1(totalLocalPoints, 0.0);
        thrust::device_vector<double> d_u2(totalLocalPoints, 0.0);
        thrust::device_vector<double> d_laplacian(totalLocalPoints, 0.0);

        dim3 blockDim(numThreads, 1, 1);
        dim3 gridDim((localX + blockDim.x - 1) / blockDim.x,
                    (localY + blockDim.y - 1) / blockDim.y,
                    (localZ + blockDim.z - 1) / blockDim.z);


        initializeField<<<gridDim, blockDim>>>(
            thrust::raw_pointer_cast(d_u0.data()),
            thrust::raw_pointer_cast(d_x.data()),
            thrust::raw_pointer_cast(d_y.data()),
            thrust::raw_pointer_cast(d_z.data()),
            Lx,
            localX, localY, localZ, 0.0
        );
        cudaCheckError(cudaGetLastError());
        cudaCheckError(cudaDeviceSynchronize());

        double startTime = MPI_Wtime();

        auto communicateBoundaryLayers = [&](const thrust::host_vector<double>& h_u) {
            for(int j = 0; j < localY; j++)
                for(int k = 0; k < localZ; k++) {
                    sendLeftX[j * localZ + k] = h_u[getIndex(0, j, k, localY, localZ)];
                    sendRightX[j * localZ + k] = h_u[getIndex(localX - 1, j, k, localY, localZ)];
                }

            for(int i = 0; i < localX; i++)
                for(int k = 0; k < localZ; k++) {
                    sendLeftY[i * localZ + k] = h_u[getIndex(i, 0, k, localY, localZ)];
                    sendRightY[i * localZ + k] = h_u[getIndex(i, localY - 1, k, localY, localZ)];
                }

            for(int i = 0; i < localX; i++)
                for(int j = 0; j < localY; j++) {
                    sendLeftZ[i * localY + j] = h_u[getIndex(i, j, 0, localY, localZ)];
                    sendRightZ[i * localY + j] = h_u[getIndex(i, j, localZ - 1, localY, localZ)];
                }

            MPI_Request requests[12];
            int requestCount = 0;

            if (neighborRanksPrev[0] != MPI_PROC_NULL) {
                MPI_Irecv(recvLeftX.data(), layerSizeYZ, MPI_DOUBLE, neighborRanksPrev[0], 0, comm, &requests[requestCount++]);
                MPI_Isend(sendLeftX.data(), layerSizeYZ, MPI_DOUBLE, neighborRanksPrev[0], 1, comm, &requests[requestCount++]);
            }
            if (neighborRanksNext[0] != MPI_PROC_NULL) {
                MPI_Irecv(recvRightX.data(), layerSizeYZ, MPI_DOUBLE, neighborRanksNext[0], 1, comm, &requests[requestCount++]);
                MPI_Isend(sendRightX.data(), layerSizeYZ, MPI_DOUBLE, neighborRanksNext[0], 0, comm, &requests[requestCount++]);
            }

            if (neighborRanksPrev[1] != MPI_PROC_NULL) {
                MPI_Irecv(recvLeftY.data(), layerSizeXZ, MPI_DOUBLE, neighborRanksPrev[1], 2, comm, &requests[requestCount++]);
                MPI_Isend(sendLeftY.data(), layerSizeXZ, MPI_DOUBLE, neighborRanksPrev[1], 3, comm, &requests[requestCount++]);
            }
            if (neighborRanksNext[1] != MPI_PROC_NULL) {
                MPI_Irecv(recvRightY.data(), layerSizeXZ, MPI_DOUBLE, neighborRanksNext[1], 3, comm, &requests[requestCount++]);
                MPI_Isend(sendRightY.data(), layerSizeXZ, MPI_DOUBLE, neighborRanksNext[1], 2, comm, &requests[requestCount++]);
            }

            if (neighborRanksPrev[2] != MPI_PROC_NULL) {
                MPI_Irecv(recvLeftZ.data(), layerSizeXY, MPI_DOUBLE, neighborRanksPrev[2], 4, comm, &requests[requestCount++]);
                MPI_Isend(sendLeftZ.data(), layerSizeXY, MPI_DOUBLE, neighborRanksPrev[2], 5, comm, &requests[requestCount++]);
            }
            if (neighborRanksNext[2] != MPI_PROC_NULL) {
                MPI_Irecv(recvRightZ.data(), layerSizeXY, MPI_DOUBLE, neighborRanksNext[2], 5, comm, &requests[requestCount++]);
                MPI_Isend(sendRightZ.data(), layerSizeXY, MPI_DOUBLE, neighborRanksNext[2], 4, comm, &requests[requestCount++]);
            }

            if(requestCount > 0) {
                MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);
            }

            if (neighborRanksPrev[0] != MPI_PROC_NULL || neighborRanksNext[0] != MPI_PROC_NULL) {
                if (recvLeftX_dev.size() != layerSizeYZ)
                    recvLeftX_dev.resize(layerSizeYZ);
                if (recvRightX_dev.size() != layerSizeYZ)
                    recvRightX_dev.resize(layerSizeYZ);

                thrust::copy(recvLeftX.begin(), recvLeftX.end(), recvLeftX_dev.begin());
                thrust::copy(recvRightX.begin(), recvRightX.end(), recvRightX_dev.begin());
            }
            else {
                if (recvLeftX_dev.size() != layerSizeYZ)
                    recvLeftX_dev.resize(layerSizeYZ, 0.0);
                else
                    thrust::fill(recvLeftX_dev.begin(), recvLeftX_dev.end(), 0.0);

                if (recvRightX_dev.size() != layerSizeYZ)
                    recvRightX_dev.resize(layerSizeYZ, 0.0);
                else
                    thrust::fill(recvRightX_dev.begin(), recvRightX_dev.end(), 0.0);
            }

            if (recvLeftY_dev.size() != layerSizeXZ)
                recvLeftY_dev.resize(layerSizeXZ);
            thrust::copy(recvLeftY.begin(), recvLeftY.end(), recvLeftY_dev.begin());

            if (recvRightY_dev.size() != layerSizeXZ)
                recvRightY_dev.resize(layerSizeXZ);
            thrust::copy(recvRightY.begin(), recvRightY.end(), recvRightY_dev.begin());

            if (recvLeftZ_dev.size() != layerSizeXY)
                recvLeftZ_dev.resize(layerSizeXY);
            thrust::copy(recvLeftZ.begin(), recvLeftZ.end(), recvLeftZ_dev.begin());

            if (recvRightZ_dev.size() != layerSizeXY)
                recvRightZ_dev.resize(layerSizeXY);
            thrust::copy(recvRightZ.begin(), recvRightZ.end(), recvRightZ_dev.begin());

            cudaCheckError(cudaDeviceSynchronize());
        };

        thrust::host_vector<double> h_u0_host = h_u0;
        thrust::copy(d_u0.begin(), d_u0.end(), h_u0_host.begin());
        communicateBoundaryLayers(h_u0_host);

        thrust::copy(d_u0.begin(), d_u0.end(), d_u1.begin());

        computeLaplacianKernel<<<gridDim, blockDim>>>(
            thrust::raw_pointer_cast(d_u0.data()),
            thrust::raw_pointer_cast(d_laplacian.data()),
            thrust::raw_pointer_cast(recvLeftX_dev.data()),
            thrust::raw_pointer_cast(recvRightX_dev.data()),
            thrust::raw_pointer_cast(recvLeftY_dev.data()),
            thrust::raw_pointer_cast(recvRightY_dev.data()),
            thrust::raw_pointer_cast(recvLeftZ_dev.data()),
            thrust::raw_pointer_cast(recvRightZ_dev.data()),
            localX, localY, localZ,
            hx * hx, hy * hy, hz * hz
        );
        cudaCheckError(cudaGetLastError());
        cudaCheckError(cudaDeviceSynchronize());

        double tau_sq_half = 0.5 * tau * tau;

        initializeU1Kernel<<<gridDim, blockDim>>>(
            thrust::raw_pointer_cast(d_u1.data()),
            thrust::raw_pointer_cast(d_u0.data()),
            thrust::raw_pointer_cast(d_laplacian.data()),
            thrust::raw_pointer_cast(d_x.data()),
            Lx,
            tau_sq_half, localX, localY, localZ
        );
        cudaCheckError(cudaGetLastError());
        cudaCheckError(cudaDeviceSynchronize());

        thrust::copy(d_u1.begin(), d_u1.end(), h_u1.begin());

        thrust::host_vector<double> h_analytical(totalLocalPoints, 0.0);
        for(int i = 0; i < localX; i++)
            for(int j = 0; j < localY; j++)
                for(int k = 0; k < localZ; k++) {
                    int idx = getIndex(i, j, k, localY, localZ);
                    h_analytical[idx] = analyticalSolution(xCoordinates[i], yCoordinates[j], zCoordinates[k], tau, Lx, Ly, Lz, at);
                }

        double localMaxError = 0.0;
        for(int idx = 0; idx < totalLocalPoints; idx++) {
            double error = fabs(h_u1[idx] - h_analytical[idx]);
            if(error > localMaxError)
                localMaxError = error;
        }

        double globalMaxError;
        MPI_Reduce(&localMaxError, &globalMaxError, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if(rank == 0)
            std::cout << "\nTime step: 0 Max err: " << globalMaxError << std::endl;

        for(int timestep = 1; timestep < Nt; timestep++) {
            thrust::copy(d_u1.begin(), d_u1.end(), h_u1.begin());
            communicateBoundaryLayers(h_u1);

            computeLaplacianKernel<<<gridDim, blockDim>>>(
                thrust::raw_pointer_cast(d_u1.data()),
                thrust::raw_pointer_cast(d_laplacian.data()),
                thrust::raw_pointer_cast(recvLeftX_dev.data()),
                thrust::raw_pointer_cast(recvRightX_dev.data()),
                thrust::raw_pointer_cast(recvLeftY_dev.data()),
                thrust::raw_pointer_cast(recvRightY_dev.data()),
                thrust::raw_pointer_cast(recvLeftZ_dev.data()),
                thrust::raw_pointer_cast(recvRightZ_dev.data()),
                localX, localY, localZ,
                hx * hx, hy * hy, hz * hz
            );
            cudaCheckError(cudaGetLastError());
            cudaCheckError(cudaDeviceSynchronize());

            updateFieldKernel<<<gridDim, blockDim>>>(
                thrust::raw_pointer_cast(d_u2.data()),
                thrust::raw_pointer_cast(d_u1.data()),
                thrust::raw_pointer_cast(d_u0.data()),
                thrust::raw_pointer_cast(d_laplacian.data()),
                thrust::raw_pointer_cast(d_x.data()),
                Lx,
                tau * tau,
                localX, localY, localZ
            );
            cudaCheckError(cudaGetLastError());
            cudaCheckError(cudaDeviceSynchronize());

            thrust::copy(d_u2.begin(), d_u2.end(), h_u2.begin());
            thrust::host_vector<double> h_analytical_step(totalLocalPoints, 0.0);
            for(int i = 0; i < localX; i++)
                for(int j = 0; j < localY; j++)
                    for(int k = 0; k < localZ; k++) {
                        int idx = getIndex(i, j, k, localY, localZ);
                        h_analytical_step[idx] = analyticalSolution(xCoordinates[i], yCoordinates[j], zCoordinates[k], tau * (timestep + 1), Lx, Ly, Lz, at);
                    }

            double localMaxErrorStep = 0.0;
            for(int idx = 0; idx < totalLocalPoints; idx++) {
                double error = fabs(h_u2[idx] - h_analytical_step[idx]);
                if(error > localMaxErrorStep)
                    localMaxErrorStep = error;
            }

            double globalMaxErrorStep;
            MPI_Reduce(&localMaxErrorStep, &globalMaxErrorStep, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
            if(rank == 0)
                std::cout << "Time step: " << timestep << " Max err: " << globalMaxErrorStep << std::endl;

            d_u0.swap(d_u1);
            d_u1.swap(d_u2);
        }

        double endTime = MPI_Wtime();
        double elapsedTime = endTime - startTime;
        double maxElapsedTime;

        MPI_Reduce(&elapsedTime, &maxElapsedTime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if(rank == 0)
            std::cout << "\nTotal simulation time: " << maxElapsedTime << " seconds" << std::endl;
    }

    cudaCheckError(cudaDeviceReset());

    MPI_Finalize();

    return 0;
}
