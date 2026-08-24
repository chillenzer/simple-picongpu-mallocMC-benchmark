# Copyright PIConGPU developers
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of PIConGPU.
#
# PIConGPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PIConGPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIConGPU.
# If not, see <http://www.gnu.org/licenses/>.
#/

# Name and Path of this Script ############################### (DO NOT change!)
export PIC_PROFILE=$(cd $(dirname ${BASH_SOURCE:-$0}) && pwd)/$(basename ${BASH_SOURCE:-$0}) # for compatibility with both zsh and bash

# General modules #############################################################
#
module purge
module load volta

# these are not yet fully exposed modules
# TODO: this needs to be updated in the future
ml use /data/rosi/shared/eb/easybuild/volta/modules/all/Core/*
ml nvidia-compilers/25.1-CUDA-12.6.0
ml OpenMPI/5.0.3

export LD_LIBRARY_PATH=/data/rosi/shared/eb/easybuild/volta/software/GCCcore/13.3.0/lib64:$LD_LIBRARY_PATH
export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=self,smcuda,vader

module load python/3.12.4
module load cmake/4.0.3
module load zlib/1.2.13-GCCcore-12.3.0

# Selfbuild software #############################################################
#

export ROSI_LIB="/bigdata/hplsim/development/rosi-picongpu-libs/"

BOOST_VERSION=1.87.0
export BOOST_ROOT=$ROSI_LIB/BOOST/$BOOST_VERSION
export CPATH=$BOOST_ROOT/include:$CPATH
export LD_LIBRARY_PATH=$BOOST_ROOT/lib:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=$BOOST_ROOT/lib/cmake:$CMAKE_PREFIX_PATH


BLOSC_VERSION=2.22.0
export BLOSC_ROOT=$ROSI_LIB/BLOSC/$BLOSC_VERSION
export CMAKE_PREFIX_PATH=$BLOSC_ROOT:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$BLOSC_ROOT/lib:$LD_LIBRARY_PATH


HDF5_VERSION=2.0.0 #1.14.6
export HDF5_ROOT=$ROSI_LIB/HDF5/$HDF5_VERSION
export PATH=$HDF5_ROOT/bin:$PATH
export CMAKE_PREFIX_PATH=$HDF5_ROOT:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$HDF5_ROOT/lib:$LD_LIBRARY_PATH


ADIOS2_VERSION=2.11.0
export ADIOS2_ROOT=$ROSI_LIB/ADIOS2/$ADIOS2_VERSION
export PATH=$ADIOS2_ROOT/bin:$PATH
export CMAKE_PREFIX_PATH=$ADIOS2_ROOT:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$ADIOS2_ROOT/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$ADIOS2_ROOT/lib/python3.10/site-packages:$PYTHONPATH


OPENPMD_VERSION=0.17.0
export OPENPMD_ROOT=$ROSI_LIB/OPENPMD/$OPENPMD_VERSION
export PATH=$OPENPMD_ROOT/bin:$PATH
export CMAKE_PREFIX_PATH=$OPENPMD_ROOT:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$OPENPMD_ROOT/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$OPENPMD_ROOT/lib/python3.10/site-packages:$PYTHONPATH


LIBPNG_VERSION=1.6.34
export LIBPNG_ROOT=$ROSI_LIB/libpng/$LIBPNG_VERSION
export CMAKE_PREFIX_PATH=$LIBPNG_ROOT:$CMAKE_PREFIX_PATH
export CPATH=$LIBPNG_ROOT/include:$CPATH
export LD_LIBRARY_PATH=$LIBPNG_ROOT/lib:$LD_LIBRARY_PATH


PNGWRITER_VERSION=0.7.0
export PNGwriter_ROOT=$ROSI_LIB/PNGWRITER/$PNGWRITER_VERSION
export CMAKE_PREFIX_PATH=$PNGwriter_ROOT:$CMAKE_PREFIX_PATH
export CPATH=$PNGwriter_ROOT/include:$CPATH
export LD_LIBRARY_PATH=$PNGwriter_ROOT/lib:$LD_LIBRARY_PATH


FFTW_VERSION=3.3.10
export FFTW_ROOT=$ROSI_LIB/FFTW/$FFTW_VERSION
export CMAKE_PREFIX_PATH=$FFTW_ROOT:$CMAKE_PREFIX_PATH
export CPATH=$FFTW_ROOT/include:$CPATH
export LD_LIBRARY_PATH=$FFTW_ROOT/lib:$LD_LIBRARY_PATH



# Environment #################################################################
#
export CC="$(which cc)"
export CXX="$(which CC)"
export CUDACXX=$(which nvcc)

export MPI_CXX=$(which mpic++)
export MPI_CC=$(which mpicc)

export PICSRC=$HOME/src/picongpu/
export PIC_EXAMPLES=$PICSRC/share/picongpu/examples
export PIC_BACKEND="cuda:70"

export PIC_SYSTEM_TEMPLATE_PATH=${PIC_SYSTEM_TEMPLATE_PATH:-"etc/picongpu/rosi-hzdr"}

export PATH=$PATH:$PICSRC
export PATH=$PATH:$PICSRC/bin
export PATH=$PATH:$PICSRC/src/tools/bin
