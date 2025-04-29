#!/bin/bash

cmake -C ../cmake/presets/most.cmake \
-D BUILD_MPI=yes \
-D LAMMPS_MACHINE=mpi \
-D PKG_REPLICA=yes \
-D PYTHON=yes \
-D PKG_PLUMED=yes \
-D PLUMED_MODE=shared \
-D DOWNLOAD_PLUMED=no \
-D PYTHON_EXECUTABLE=/home/adam/anaconda3/envs/lmp/bin/python \
-D LAMMPS_INSTALL_RPATH=on \
-D BUILD_SHARED_LIBS=on \
../cmake






