#!/bin/bash
#SBATCH -N 50
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -J aSiO2
#SBATCH -t 10:00:00

#SBATCH --output=/global/cscratch1/sd/agoga/slurm-output/SiO-%j.txt
#SBATCH --mail-user="adgoga@ucdavis.edu"
#SBATCH --mail-type=FAIL,END

#COMMAND LINE ARGUMENTS
#1ST FILE NAME
#2ND 'farm' or no to tell if farm or not
#3RD 


FILENAME=$1

NAME=${FILENAME%.*}
UNIQUE_TAG=$(date +%m%d-%Hh%Mm%S)
CWD=$(pwd) #current working directory
OUT_FOLDER=$CWD"/output/"${NAME}${UNIQUE_TAG}"/"
mkdir -p $CWD"/output/" #just in case output folder is not made
mkdir $OUT_FOLDER #Now make folder where all the output will go


IN_FILE=$CWD"/"$FILENAME
LOG_FILE=$OUT_FOLDER$NAME".log"
cp $IN_FILE $OUT_FOLDER

s=$OUT_FOLDER$NAME"_SLURM.txt"

module load cray-fftw
module load tbb
module load openmpi


j=$SLURM_JOB_ID

export OMP_NUM_THREADS=2
export OMP_PLACES=threads
export OMP_PROC_BIND=true
#                                                           Creates a variable in lammps ${output_folder}
srun -n 1600 -c 2 --cpu_bind=cores $HOME/lmp_haswell -nocite -log $LOG_FILE -in $IN_FILE -var output_folder $OUT_FOLDER
