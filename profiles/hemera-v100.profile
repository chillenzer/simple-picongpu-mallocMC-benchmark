# Copyright 
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
export PIC_PROFILE=$(cd $(dirname $BASH_SOURCE) && pwd)"/"$(basename $BASH_SOURCE)

# General modules #############################################################
#
module purge
module load git
module load gcc/12.2.0
module load cuda/12.1
module load libfabric/1.17.0
module load ucx/1.14.0-gdr
module load openmpi/4.1.5-cuda121-gdr
module load python/3.10.4
module load boost/1.82.0

# Other Software ##############################################################
#
module load zlib/1.2.11
module load hdf5-parallel/1.12.0-omp415-cuda121
module load c-blosc/1.21.4
module load adios2/2.9.2-cuda121
module load openpmd/0.15.2-cuda121
module load cmake/3.26.1
module load fftw/3.3.10-ompi415-cuda121-gdr

module load libpng/1.6.39
module load pngwriter/0.7.0

# Environment #################################################################
#
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$BOOST_LIB

export PICSRC=<to be completed>
export PIC_EXAMPLES=$PICSRC/share/picongpu/examples
export PIC_BACKEND="cuda:70"

# Path to the required templates of the system,
# relative to the PIConGPU source code of the tool bin/pic-create.
export PIC_SYSTEM_TEMPLATE_PATH=${PIC_SYSTEM_TEMPLATE_PATH:-"etc/picongpu/hemera-hzdr"}

export PATH=$PICSRC/bin:$PATH
export PATH=$PICSRC/src/tools/bin:$PATH

export PYTHONPATH=$PICSRC/lib/python:$PYTHONPATH
