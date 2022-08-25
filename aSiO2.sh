#!/bin/bash -l
#! -cwd
#! -j y
#! -S /bin/bash
#SBATCH -D ./
#SBATCH --job-name=a-SiO2
#SBATCH --partition=med2 # Partition you are running on. Options: low2, med2, high2
#SBATCH --output=/home/agoga/sandbox/lammps/topcon/slurm-output/SiO-%j.txt
#SBATCH --mail-user="adgoga@ucdavis.edu"
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#======memory for neb I will specifically set up a grid nxm makes n-replicas that use m sub-tasks ntasks = n*m
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=32 
#SBATCH --cpus-per-task=1 
#SBATCH --mem=64G
#SBATCH -t 3-00:00
#blahhh SBATCH --array=0-1

# Here we will run a hybrid MPI/OPENMP code. E.g. we have multiple tasks, and each task has multiple threads
export OMP_NUM_THREADS=1
j=$SLURM_JOB_ID
#i=$SLURM_ARRAY_TASK_ID
FOLDER=$HOME"/sandbox/lammps/topcon/"
LFILE=test3.log
INFILE=$FOLDER"SilicaAnneal.lmp"
PART=16
srun /home/agoga/sandbox/lammps/lmp_mpi -log $LFILE -in $INFILE
