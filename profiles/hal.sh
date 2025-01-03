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

# User Information ################################# (edit the following lines)
#   - automatically add your name and contact to output file meta data
#   - send me a mail on batch system jobs: NONE, BEGIN, END, FAIL, REQUEUE, ALL,
#     TIME_LIMIT, TIME_LIMIT_90, TIME_LIMIT_80 and/or TIME_LIMIT_50
export MY_MAILNOTIFY="ALL"
export MY_MAIL="someone@example.com"
export MY_NAME="$(whoami) <$MY_MAIL>"

# Text Editor for Tools ###################################### (edit this line)
#   - examples: "nano", "vim", "emacs -nw", "vi" or without terminal: "gedit"
export EDITOR="vim"

# load packages
spack unload

# PIConGPU build dependencies #################################################
#   need to load correct cmake and gcc to compile picongpu

spack load gcc@12.2.0
spack load cmake@3.26.6 ^openssl certs=mozilla %gcc@12.2.0

# General modules #############################################################
#   correct dependencies are automatically loaded, if successfully installed using install.sh
#   and no name confilcts in spack, see install.sh for more precise definition
#   if name conflicts occur

spack load openpmd-api@0.15.2 %gcc@12.2.0 \
    ^adios2@2.9.2 \
    ^hdf5@1.14.3 \
    ^openmpi@4.1.5 +atomics +cuda cuda_arch=80 \
    ^python@3.11.6 \
    ^py-numpy@1.26.2
spack load boost@1.83.0 %gcc@12.2.0

# PIConGPU output dependencies ################################################
#
spack load pngwriter@0.7.0 %gcc@12.2.0

# Python pip dependency #######################################################
spack load py-pip ^python@3.11.6 %gcc@12.2.0


# Environment #################################################################
#
export PICSRC=<to be completed>
export PIC_EXAMPLES=$PICSRC/share/picongpu/examples
export PIC_BACKEND="cuda:80"

# Path to the required templates of the system,
# relative to the PIConGPU source code of the tool bin/pic-create.
export PIC_SYSTEM_TEMPLATE_PATH=${PIC_SYSTEM_TEMPLATE_PATH:-"etc/picongpu/bash-devServer-hzdr"}

export PATH=$PICSRC/bin:$PATH
export PATH=$PICSRC/src/tools/bin:$PATH

export PYTHONPATH=$PICSRC/lib/python:$PYTHONPATH

# "tbg" default options #######################################################
export TBG_SUBMIT="bash"
export TBG_TPLFILE="etc/picongpu/bash-devServer-hzdr/mpiexec.tpl"

# Load autocompletion for PIConGPU commands
BASH_COMP_FILE=$PICSRC/bin/picongpu-completion.bash
if [ -f "$BASH_COMP_FILE" ] ; then
    source $BASH_COMP_FILE
else
    echo "bash completion file '$BASH_COMP_FILE' not found." >&2
fi
