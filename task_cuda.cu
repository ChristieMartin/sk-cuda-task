#include <mpi.h>
#include <fstream>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

const int numThreads = 128;

struct Buffer {
    int xLeft;
    int xRight;
    int yLeft;
    int yRight;
    int zLeft;
    int zRight;
    int xLen;
    int yLen;
    int zLen;
    int size;

    Buffer() {}

    Buffer(int xLeft, int xRight, int yLeft, int yRight, int zLeft, int zRight) : xLeft(xLeft), xRight(xRight),
        yLeft(yLeft), yRight(yRight), zLeft(zLeft), zRight(zRight) {
        xLen = xRight - xLeft + 1;
        yLen = yRight - yLeft + 1;
        zLen = zRight - zLeft + 1;
        size = xLen * yLen * zLen;
    }

    inline bool operator<(const Buffer& other) const {
        return other.xLeft <= xLeft && xRight <= other.xRight &&
               other.yLeft <= yLeft && yRight <= other.yRight &&
               other.zLeft <= zLeft && zRight <= other.zRight;
    }
};


void initializeBuffers(std::vector<Buffer> &buffers, int dim, int xLeft, int xRight, int yLeft, int yRight, int zLeft, int zRight, int size) {
    
    if (size == 1) {
        buffers.push_back(Buffer(xLeft, xRight, yLeft, yRight, zLeft, zRight));
        return;
    }

    if (size % 2 == 1) { 
        if (dim == 0) {
            int x = xLeft + (xRight - xLeft) / size;
            buffers.push_back(Buffer(xLeft, x, yLeft, yRight, zLeft, zRight));
            xLeft = x + 1;
            dim = 1;
        }
        else if (dim == 1) {
            int y = yLeft + (yRight - yLeft) / size;
            buffers.push_back(Buffer(xLeft, xRight, yLeft, y, zLeft, zRight));
            yLeft = y + 1;
            dim = 2;
        }
        else { 
            int z = zLeft + (zRight - zLeft) / size;
            buffers.push_back(Buffer(xLeft, xRight, yLeft, yRight, zLeft, z));
            zLeft = z + 1;
            dim = 0;
        }

        size--;
    }

    
    if (dim == 0) {
        int x = (xLeft + xRight) / 2;
        initializeBuffers(buffers, 1, xLeft, x, yLeft, yRight, zLeft, zRight, size / 2);
        initializeBuffers(buffers, 1, x + 1, xRight, yLeft, yRight, zLeft, zRight, size / 2);
    }
    else if (dim == 1) {
        int y = (yLeft + yRight) / 2;
        initializeBuffers(buffers, 2, xLeft, xRight, yLeft, y, zLeft, zRight, size / 2);
        initializeBuffers(buffers, 2, xLeft, xRight, y + 1, yRight, zLeft, zRight, size / 2);
    }
    else {
        int z = (zLeft + zRight) / 2;
        initializeBuffers(buffers, 0, xLeft, xRight, yLeft, yRight, zLeft, z, size / 2);
        initializeBuffers(buffers, 0, xLeft, xRight, yLeft, yRight, z + 1, zRight, size / 2);
    }
}

__device__ double analyticalSolution(double x, double y, double z, double t, double at, double Lx, double Ly, double Lz) {
    return sin(3 * M_PI * x / Lx) * sin(2 * M_PI * y / Ly) * sin(2 * M_PI * z / Lz) * cos(at * t + 4 * M_PI);
}


__device__ double phi(double x, double y, double z, double at, double Lx, double Ly, double Lz ) {
    return analyticalSolution(x, y, z, 0, at, Lx, Ly, Lz);
}


__host__ __device__ inline int getIndex(int i, int j, int k, const Buffer buffer) {
    return (i - buffer.xLeft) * buffer.yLen * buffer.zLen + (j - buffer.yLeft) * buffer.zLen + (k - buffer.zLeft);
}

__global__ void initializeField(double *u0, double at, int size, int x1, int y1, int z1, int yLen, int zLen, double hx, double hy, double hz, double Lx, double Ly, double Lz, const Buffer buffer) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size)
        return;

    int i = x1 + index / (yLen * zLen);
    int j = y1 + index % (yLen * zLen) / zLen;
    int k = z1 + index % zLen;

    u0[getIndex(i, j, k, buffer)] = phi(i * hx, j * hy, k * hz, at, Lx, Ly, Lz);
}

__device__ double getRecvValue(double *u, int i, int j, int k, const Buffer buffer, double *recv, Buffer *recvBuffers, int bufferSize) {
    if (buffer.xLeft <= i && i <= buffer.xRight && buffer.yLeft <= j && j <= buffer.yRight && buffer.zLeft <= k && k <= buffer.zRight) {
        return u[getIndex(i, j, k, buffer)];
    }

    int offset = 0;

    for (int r_i = 0; r_i < bufferSize; r_i++) {
        Buffer recvBuffer = recvBuffers[r_i];

        if (i < recvBuffer.xLeft || i > recvBuffer.xRight || j < recvBuffer.yLeft || j > recvBuffer.yRight || k < recvBuffer.zLeft || k > recvBuffer.zRight) {
            offset += recvBuffers[r_i].size;
            continue;
        }
        return recv[offset + getIndex(i, j, k, recvBuffer)];
    }
    return 0;
}

__device__ double computeLaplacianKernel(double *u, int i, int j, int k, const Buffer buffer, double hx, double hy, double hz, double *recv, Buffer *recvBuffers, int bufferSize) {
    double dx = (getRecvValue(u, i, j - 1, k, buffer, recv, recvBuffers, bufferSize) - 2 * u[getIndex(i, j, k, buffer)] + getRecvValue(u, i, j + 1, k, buffer, recv, recvBuffers, bufferSize)) / (hy * hy);
    double dy = (getRecvValue(u, i - 1, j, k, buffer,  recv, recvBuffers, bufferSize) - 2 * u[getIndex(i, j, k, buffer)] + getRecvValue(u, i + 1, j, k, buffer, recv, recvBuffers, bufferSize)) / (hx * hx);
    double dz = (getRecvValue(u,i, j, k - 1, buffer, recv, recvBuffers, bufferSize) - 2 * u[getIndex(i, j, k, buffer)] + getRecvValue(u, i, j, k + 1, buffer, recv, recvBuffers, bufferSize)) / (hz * hz);
    return dx + dy + dz;
}

__global__ void updateFieldKernel(double *u, double *u0, double *u1, double *recv, Buffer *recvBuffers, int bufferSize,
                               int size, int x1, int y1, int z1, int yLen, int zLen, double tau, double hx, double hy, double hz, const Buffer buffer) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size)
        return;

    int i = x1 + index / (yLen * zLen);
    int j = y1 + index % (yLen * zLen) / zLen;
    int k = z1 + index % zLen;

    u[getIndex(i, j, k, buffer)] = 2 * u1[getIndex(i, j, k, buffer)] - u0[getIndex(i, j, k, buffer)] +
            tau * tau * computeLaplacianKernel(u1, i, j, k, buffer, hx, hy, hz, recv, recvBuffers, bufferSize);
}

__global__ void initializeU1Kernel(double *u0, double *u1, double *recv, Buffer *recvBuffers, int bufferSize,
                                     int size, int x1, int y1, int z1, int yLen, int zLen, double tau, double hx, double hy, double hz, const Buffer buffer) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size)
        return;

    int i = x1 + index / (yLen * zLen);
    int j = y1 + index % (yLen * zLen) / zLen;
    int k = z1 + index % zLen;

    u1[getIndex(i, j, k, buffer)] = u0[getIndex(i, j, k, buffer)] + tau * tau / 2 * computeLaplacianKernel(u0, i, j, k, buffer, hx, hy, hz, recv, recvBuffers, bufferSize);
}

thrust::host_vector<double> getSendData(thrust::host_vector<double> &u, const Buffer buffer, const Buffer neighbourBuffer) {
    thrust::host_vector<double> sendData(neighbourBuffer.size);

    for (int i = neighbourBuffer.xLeft; i <= neighbourBuffer.xRight; i++)
        for (int j = neighbourBuffer.yLeft; j <= neighbourBuffer.yRight; j++)
            for (int k = neighbourBuffer.zLeft; k <= neighbourBuffer.zRight; k++)
                sendData[getIndex(i, j, k, neighbourBuffer)] = u[getIndex(i, j, k, buffer)];

    return sendData;
}

thrust::host_vector<double> communicateBetweenLayers(thrust::host_vector<double> &u, const Buffer buffer,
                                     thrust::host_vector<Buffer> &sendBuffers, thrust::host_vector<Buffer> &recvBuffers, thrust::host_vector<int> &neighboursRanks) {
    thrust::host_vector<double> recvData;
    int offset = 0;
    thrust::host_vector<MPI_Request> requests(2);
    thrust::host_vector<MPI_Status> statuses(2);

    for (int i = 0; i < neighboursRanks.size(); i++) {
        thrust::host_vector<double> sendData = getSendData(u, buffer, sendBuffers[i]);
        recvData.insert(recvData.end(), recvBuffers[i].size, 0);

        MPI_Isend(sendData.data(), sendBuffers[i].size, MPI_DOUBLE, neighboursRanks[i], 0, MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(recvData.data() + offset, recvBuffers[i].size, MPI_DOUBLE, neighboursRanks[i], 0, MPI_COMM_WORLD, &requests[1]);
        MPI_Waitall(2, requests.data(), statuses.data());
        offset += recvBuffers[i].size;
    }
    return recvData;
}

__global__ void computeLayerErrorKernel(double *u, double t, double at, const Buffer buffer, double hx, double hy, double hz, double Lx, double Ly, double Lz) {
    int idx = threadIdx.x;

    for (int index = idx; index < buffer.size; index += numThreads) {
        int i = buffer.xLeft + index / (buffer.yLen * buffer.zLen);
        int j = buffer.yLeft + index % (buffer.yLen * buffer.zLen) / buffer.zLen;
        int k = buffer.zLeft + index % buffer.zLen;

        u[getIndex(i, j, k, buffer)] = fabs(u[getIndex(i, j, k, buffer)] - analyticalSolution(i * hx, j * hy, k * hz, t, at, Lx, Ly, Lz));
    }
}

double computeMaximumError(thrust::device_vector<double> &targetU, double t, const Buffer buffer, double hx, double hy, double hz, double Lx, double Ly, double Lz, double at) {
    computeLayerErrorKernel<<<1, numThreads>>>(thrust::raw_pointer_cast(&targetU[0]), t, at, buffer, hx, hy, hz, Lx, Ly, Lz);
    thrust::device_vector<double>::iterator iter = thrust::max_element(targetU.begin(), targetU.end());

    double localMaxError = targetU[iter - targetU.begin()];
    double globalMaxError;

    MPI_Reduce(&localMaxError, &globalMaxError, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    return globalMaxError;
}


__global__ void communicateBoundaryLayer(double *u, const Buffer buffer, double hx, double hy, double hz, double Lx, double Ly, double Lz, double at, double tau, int dim, int border, int iLeft, int jLeft, int iLen, int jLen, bool periodic) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= iLen * jLen)
        return;

    int i = iLeft + index / jLen;
    int j = jLeft + index % jLen;

    if (dim == 0) {
        if (periodic) {
            u[getIndex(border, i, j, buffer)] = analyticalSolution(border * hx, i * hy, j * hz, tau, at, Lx, Ly, Lz);
        } else {
            u[getIndex(border, i, j, buffer)] = 0;
        }
    } else if (dim == 1) {
        if (periodic) {
            u[getIndex(i, border, j, buffer)] = analyticalSolution(i * hx, border * hy, j * hz, tau, at, Lx, Ly, Lz);
        } else {
            u[getIndex(i, border, j, buffer)] = 0;
        }
    } else {
        if (periodic) {
            u[getIndex(i, j, border, buffer)] = analyticalSolution(i * hx, j * hy, border * hz, tau, at, Lx, Ly, Lz);
        } else {
            u[getIndex(i, j, border, buffer)] = 0;
        }
    }
}

void communicateBoundaryLayers(thrust::device_vector<double> &targetU, double tau, const Buffer buffer, int N, double hx, double hy, double hz, double Lx, double Ly, double Lz, double at) {
    
    if (buffer.xLeft == 0) {
        communicateBoundaryLayer<<<((buffer.yLen * buffer.zLen + numThreads - 1) / numThreads), numThreads>>>(thrust::raw_pointer_cast(&targetU[0]), buffer, hx, hy, hz, Lx, Ly, Lz, at, tau, 0, 0, buffer.yLeft, buffer.zLeft, buffer.yLen, buffer.zLen, false);
    }

    if (buffer.xRight == N) {
        communicateBoundaryLayer<<<((buffer.yLen * buffer.zLen + numThreads - 1) / numThreads), numThreads>>>(thrust::raw_pointer_cast(&targetU[0]), buffer, hx, hy, hz, Lx, Ly, Lz, at, tau, 0, N, buffer.yLeft, buffer.zLeft, buffer.yLen, buffer.zLen, false);
    }

    if (buffer.yLeft == 0) {
        communicateBoundaryLayer<<<((buffer.xLen * buffer.zLen + numThreads - 1) / numThreads), numThreads>>>(thrust::raw_pointer_cast(&targetU[0]), buffer, hx, hy, hz, Lx, Ly, Lz, at, tau, 1, 0, buffer.xLeft, buffer.zLeft, buffer.xLen, buffer.zLen, true);
    }

    if (buffer.yRight == N) {
        communicateBoundaryLayer<<<((buffer.xLen * buffer.zLen + numThreads - 1) / numThreads), numThreads>>>(thrust::raw_pointer_cast(&targetU[0]), buffer, hx, hy, hz, Lx, Ly, Lz, at, tau, 1, N, buffer.xLeft, buffer.zLeft, buffer.xLen, buffer.zLen, true);
    }

    if (buffer.zLeft == 0) {
        communicateBoundaryLayer<<<((buffer.xLen * buffer.yLen + numThreads - 1) / numThreads), numThreads>>>(thrust::raw_pointer_cast(&targetU[0]), buffer, hx, hy, hz, Lx, Ly, Lz, at, tau, 2, 0, buffer.xLeft, buffer.yLeft, buffer.xLen, buffer.yLen, true);
    }

    if (buffer.zRight == N) {
        communicateBoundaryLayer<<<((buffer.xLen * buffer.yLen + numThreads - 1) / numThreads), numThreads>>>(thrust::raw_pointer_cast(&targetU[0]), buffer, hx, hy, hz, Lx, Ly, Lz, at, tau, 2, N, buffer.xLeft, buffer.yLeft, buffer.xLen, buffer.yLen, true);
    }
}

void getNeighbours(const std::vector<Buffer> &buffers, thrust::host_vector<Buffer> &sendBuffers, thrust::host_vector<Buffer> &recvBuffers, thrust::host_vector<int> &neighboursRanks, int rank, int size) {
    Buffer buffer = buffers[rank];

    for (int i = 0; i < size; i++) {
        if (i == rank)
            continue;

        Buffer neighbourBuffer = buffers[i];
        if (buffer.xLeft == neighbourBuffer.xRight + 1 || neighbourBuffer.xLeft == buffer.xRight + 1) {
            int xSend = buffer.xLeft == neighbourBuffer.xRight + 1 ? buffer.xLeft : buffer.xRight;
            int xRecv = neighbourBuffer.xLeft == buffer.xRight + 1 ? neighbourBuffer.xLeft : neighbourBuffer.xRight;
            int yLeft, yRight, zLeft, zRight;

            if (buffer < neighbourBuffer) {
                yLeft = buffer.yLeft; 
                yRight = buffer.yRight; 
                zLeft = buffer.zLeft; 
                zRight = buffer.zRight;
            } else if (neighbourBuffer < buffer) {
                yLeft = neighbourBuffer.yLeft; 
                yRight = neighbourBuffer.yRight; 
                zLeft = neighbourBuffer.zLeft; 
                zRight = neighbourBuffer.zRight;
            } else {
                continue;
            }
            
            sendBuffers.push_back(Buffer(xSend, xSend, yLeft, yRight, zLeft, zRight));
            recvBuffers.push_back(Buffer(xRecv, xRecv, yLeft, yRight, zLeft, zRight));
            neighboursRanks.push_back(i);
        } else
        if (buffer.yLeft == neighbourBuffer.yRight + 1 || neighbourBuffer.yLeft == buffer.yRight + 1) {
            int ySend = buffer.yLeft == neighbourBuffer.yRight + 1 ? buffer.yLeft : buffer.yRight;
            int yRecv = neighbourBuffer.yLeft == buffer.yRight + 1 ? neighbourBuffer.yLeft : neighbourBuffer.yRight;
            int xLeft, xRight, zLeft, zRight;

            if (buffer < neighbourBuffer) {
                xLeft = buffer.xLeft; 
                xRight = buffer.xRight; 
                zLeft = buffer.zLeft; 
                zRight = buffer.zRight;
            } else if (neighbourBuffer < buffer) {
                xLeft = neighbourBuffer.xLeft; 
                xRight = neighbourBuffer.xRight; 
                zLeft = neighbourBuffer.zLeft; 
                zRight = neighbourBuffer.zRight;
            } else {
                continue;
            }
            sendBuffers.push_back(Buffer(xLeft, xRight, ySend, ySend, zLeft, zRight));
            recvBuffers.push_back(Buffer(xLeft, xRight, yRecv, yRecv, zLeft, zRight));
            neighboursRanks.push_back(i);
        } else
        if (buffer.zLeft == neighbourBuffer.zRight + 1 || neighbourBuffer.zLeft == buffer.zRight + 1) {
            int zSend = buffer.zLeft == neighbourBuffer.zRight + 1 ? buffer.zLeft : buffer.zRight;
            int zRecv = neighbourBuffer.zLeft == buffer.zRight + 1 ? neighbourBuffer.zLeft : neighbourBuffer.zRight;
            int xLeft, xRight, yLeft, yRight;

            if (buffer < neighbourBuffer) {
                xLeft = buffer.xLeft; 
                xRight = buffer.xRight; 
                yLeft = buffer.yLeft; 
                yRight = buffer.yRight;
            } else if (neighbourBuffer < buffer) {
                xLeft = neighbourBuffer.xLeft; 
                xRight = neighbourBuffer.xRight; 
                yLeft = neighbourBuffer.yLeft; 
                yRight = neighbourBuffer.yRight;
            } else {
                continue;
            }
            sendBuffers.push_back(Buffer(xLeft, xRight, yLeft, yRight, zSend, zSend));
            recvBuffers.push_back(Buffer(xLeft, xRight, yLeft, yRight, zRecv, zRecv));
            neighboursRanks.push_back(i);
        }
    }
}

int main(int argc, char** argv) {
    
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 128;
    double T = 0.001;
    int Nt = 20;
    double tau = T / Nt;

    double L = 1.0;
    double hx = 0.0, hy = 0.0, hz = 0.0;
    double at = 0.0;

    if (argc > 1) {
        try {
            double grid_length = std::stod(argv[1]);
            N = static_cast<int>(grid_length);
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
            L = M_PI;
            if(rank == 0) std::cout << "Domain length set to pi in each dimension." << std::endl;
        } else {
            try {
                double length = std::stod(arg);
                L = length;
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

    double startTime = MPI_Wtime();

    double Lx = L;
    double Ly = L;
    double Lz = L;

    hx = Lx / N;
    hy = Ly / N;
    hz = Lz / N;

    tau = T / Nt;

    at = M_PI * sqrt(9.0 / (Lx * Lx) + 4.0 / (Ly * Ly) + 4.0 / (Lz * Lz));

    std::vector<Buffer> buffers;
    initializeBuffers(buffers, 0, 0, N, 0, N, 0, N, size);
    Buffer buffer = buffers[rank];

    std::vector< thrust::device_vector<double> > u(3);
    for (int i = 0; i < 3; i++)
        u[i].resize(buffer.size);

    thrust::host_vector<Buffer> sendBuffers, recvBuffers;
    thrust::host_vector<int> neighboursRanks;
    getNeighbours(buffers, sendBuffers, recvBuffers, neighboursRanks, rank, size);

    communicateBoundaryLayers(u[0], 0, buffer, N, hx, hy, hz, Lx, Ly, Lz, at);
    communicateBoundaryLayers(u[1], tau, buffer, N, hx, hy, hz, Lx, Ly, Lz, at);

    int x1 = std::max(buffer.xLeft, 1); int x2 = std::min(buffer.xRight, N - 1);
    int y1 = std::max(buffer.yLeft, 1); int y2 = std::min(buffer.yRight, N - 1);
    int z1 = std::max(buffer.zLeft, 1); int z2 = std::min(buffer.zRight, N - 1);
    
    int localX = x2 - x1 + 1;
    int localY = y2 - y1 + 1;
    int localZ = z2 - z1 + 1;
    int layerSize = localX * localY * localZ;

    initializeField<<<(layerSize + numThreads - 1) / numThreads, numThreads>>>(thrust::raw_pointer_cast(&u[0][0]), at, layerSize, x1, y1, z1, localY, localZ, hx, hy, hz, Lx, Ly, Lz, buffer);

    thrust::host_vector<double> u0(u[0]);
    thrust::host_vector<double> recv = communicateBetweenLayers(u0, buffer, sendBuffers, recvBuffers, neighboursRanks);
    thrust::device_vector<double> recvData(recv);
    thrust::device_vector<Buffer> recvBuffer(recvBuffers);

    initializeU1Kernel<<<(layerSize + numThreads - 1) / numThreads, numThreads>>>(thrust::raw_pointer_cast(&u[0][0]), thrust::raw_pointer_cast(&u[1][0]), thrust::raw_pointer_cast(&recvData[0]), thrust::raw_pointer_cast(&recvBuffer[0]), recvBuffers.size(), layerSize, x1, y1, z1, localY, localZ, tau, hx, hy, hz, buffer);
    
    for (int timestep = 2; timestep <= Nt; timestep++) {
        thrust::host_vector<double> currentU(u[(timestep + 2) % 3]);
        recv = communicateBetweenLayers(currentU, buffer, sendBuffers, recvBuffers, neighboursRanks);
        thrust::device_vector<double> recvData(recv);
        thrust::device_vector<Buffer> recvBuffer(recvBuffers);

        updateFieldKernel<<<(layerSize + numThreads - 1) / numThreads, numThreads>>>(thrust::raw_pointer_cast(&u[timestep % 3][0]), thrust::raw_pointer_cast(&u[(timestep + 1) % 3][0]), thrust::raw_pointer_cast(&u[(timestep + 2) % 3][0]), thrust::raw_pointer_cast(&recvData[0]), thrust::raw_pointer_cast(&recvBuffer[0]), recvBuffers.size(), layerSize, x1, y1, z1, localY, localZ, tau, hx, hy, hz, buffer);
        communicateBoundaryLayers(u[timestep % 3], timestep * tau, buffer, N, hx, hy, hz, Lx, Ly, Lz, at);
    }

    double finalError = computeMaximumError(u[Nt % 3], Nt * tau, buffer, hx, hy, hz, Lx, Ly, Lz, at);
    
    double endTime = MPI_Wtime();
    double elapsedTime = endTime - startTime;
    double maxElapsedTime;
    MPI_Reduce(&elapsedTime, &maxElapsedTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\n\nTotal time: " << maxElapsedTime << " seconds, Max err: " << finalError  << std::endl; 
    }
    MPI_Finalize();
    return 0;
}