run-polus-test:
	# module load SpectrumMPI/10.1.0
	# scl enable devtoolset-9 bash
	
	nvcc -O3 -std=c++11 -arch=sm_35 --compiler-bindir /opt/ibm/spectrum_mpi/bin/mpic++ -I/opt/ibm/spectrum_mpi/include -L/opt/ibm/spectrum_mpi/lib -lmpiprofilesupport -lmpi_ibm -o task_cuda task_cuda.cu

	mkdir -p cuda_outputs

	for N in 128; do \
		for p in 1; do \
			bsub -n $$p -gpu "num=2" -R "span[ptile=2]" -W 00:10 -e err.txt -oo cuda_outputs/N$$N\_Lpi\_P$$p.txt mpiexec ./task_cuda $$N pi 0.0001 ; \
		done \
	done

run-polus-cuda:
	# module load SpectrumMPI/10.1.0
	# scl enable devtoolset-9 bash

	nvcc -O3 -std=c++11 -arch=sm_35 --compiler-bindir /opt/ibm/spectrum_mpi/bin/mpic++ -I/opt/ibm/spectrum_mpi/include -L/opt/ibm/spectrum_mpi/lib -lmpiprofilesupport -lmpi_ibm -o task_cuda task_cuda.cu

	mkdir -p cuda_outputs

	for N in 128 256 512 ; do \
		for p in 1 10 20 ; do \
			bsub -n $$p -gpu "num=2" -R "span[ptile=2]" -W 00:10 -e err.txt -oo cuda_outputs/N$$N\_L1\_P$$p.txt mpiexec ./task_cuda $$N 1 0.0001 ; \
			bsub -n $$p -gpu "num=2" -R "span[ptile=2]" -W 00:10 -e err.txt -oo cuda_outputs/N$$N\_Lpi\_P$$p.txt mpiexec ./task_cuda $$N pi 0.0001 ; \
		done \
	done

run-prof-cuda:
	# module load SpectrumMPI/10.1.0
	# scl enable devtoolset-9 bash
	nvcc -O3 -std=c++11 -arch=sm_35 --compiler-bindir /opt/ibm/spectrum_mpi/bin/mpic++ -I/opt/ibm/spectrum_mpi/include -L/opt/ibm/spectrum_mpi/lib -lmpiprofilesupport -lmpi_ibm -o task_cuda task_cuda.cu
	for N in 128 256 512 ; do \
		bsub -n 1 -gpu "num=2" -W 00:10 -o /dev/null -e profiling_err_$$N.txt -oo cuda_outputs/prof_N$$N\_L1\_P$$p.txt mpiexec /usr/local/cuda/bin/nvprof ./task4 $$N 1 0.0001 ; \
	done