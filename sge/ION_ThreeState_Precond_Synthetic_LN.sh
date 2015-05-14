#!/bin/bash
#
#
# Request 1GB RAM (virtual and total)
#$ -l h_vmem=2G,tmem=2G
#
# Specify a maximum execution time of 1 hour
#$ -l h_rt=2:00:00
#
# Perform a reservation for the resources required for this job
#$ -R y
#
# Use 10G ethernet between nodes (helps with OpenMPI errors but not all nodes have 10g e.g. stats nodes don't)
#$ -l 10g=yes
#
# Interpret paths relative to the current working directory
#$ -cwd
#
# Place job output streams (stdout and stderr) in a single file
#$ -j y
#
# Use Bash as the shell for this script
#$ -S /bin/bash
#
# Specify the name for the job
#$ -N ION_3STATE_SYNTHETIC_LN
#$ -o /home/ucbpmep/GMCMC/ION_3STATE_SYNTHETIC_LN.out
#$ -e /home/ucbpmep/GMCMC/ION_3STATE_SYNTHETIC_LN.err

# Paths to Sundials, OpenMPI and HDF5
export SUNDIALS_ROOT=${HOME}
export MPI_ROOT=/share/apps/openmpi-1.8.1
export HDF5_ROOT=/share/apps/hdf5

# Add Sundials, OpenMPI and HDF5 to path and library path
export PATH=${SUNDIALS_ROOT}/bin:${MPI_ROOT}/bin:${HDF5_ROOT}/bin:${PATH}
export LD_LIBRARY_PATH=${SUNDIALS_ROOT}/lib:${MPI_ROOT}/lib:${HDF5_ROOT}/lib:/opt/gridengine/lib/lx26-amd64:/home/ucbpmep/dcprogs/build/likelihood:/share/apps/openmpi-1.6.5/lib:/share/apps/matlabR2013a/bin/glnxa64


# Run the program using mpirun
# The mca parameter specifies to use the 10g ethernet ports on each node.
# NSLOTS is the number of cores assigned by SGE.
# Which nodes to use is passed to OpenMPI by SGE.

DATE=$( date +%s )

${MPI_ROOT}/bin/mpirun --mca btl_tcp_if_exclude lo,eth0 -n ${NSLOTS} ./ION_ThreeState_4Param_Precond_Synthetic_LN_MH results/ION_3STATE_SYNTHETIC_LN_BurnIn_${NSLOTS}_${DATE}.h5 results/ION_3STATE_SYNTHETIC_LN_Posterior_${NSLOTS}_${DATE}.h5 --precondition -t ${NSLOTS}

exit 0
