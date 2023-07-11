#!/bin/bash

module load openmpi
module load fftw
module load intel-oneapi-mkl
module load cmake 

cmake -C ../cmake/presets/most.cmake \
-D BUILD_MPI=yes \
-D LAMMPS_MACHINE=mpi \
-D PKG_REPLICA=yes \
-D BUILD_SHARED_LIBS=yes \
-D CMAKE_PREFIX_PATH=/share/apps/22.04/openmpi/4.1.5/ \
-D CMAKE_C_COMPILER=/share/apps/22.04/openmpi/4.1.5/bin/mpicc \
-D CMAKE_CXX_COMPILER=/share/apps/22.04/openmpi/4.1.5/bin/mpicxx \
-D PYTHON_EXECUTABLE=/home/agoga/.conda/envs/lmp/bin/python \
../cmake