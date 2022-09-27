#!/bin/bash -l
#! -cwd
#! -j y
#! -S /bin/bash
#SBATCH -D ./
#SBATCH --job-name=a-SiO2
#SBATCH --partition=med2 # Partition you are running on. Options: low2, med2, high2
#SBATCH --output=/home/agoga/sandbox/lammps/topcon/slurm-output/SiO-%j.txt
#SBATCH --mail-user="adgoga@ucdavis.edu"
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-type=FAIL


#======
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1 
#SBATCH --mem=64G
#SBATCH -t 4-00:00

###COMMAND LINE ARGUMENTS
###1ST FILE NAME
###2ND 'farm' or no to tell if farm or not
###3RD 


FILENAME=$1 #"SilicaAnneal.lmp"
j=$SLURM_JOB_ID
PART=16

export OMP_NUM_THREADS=1
NAME=${FILENAME%.*}

if [[ $2 == "farm" ]]; then
    UNIQUE_TAG=$SLURM_JOBID
else 
    UNIQUE_TAG=$(date +%m%d-%Hh%Mm%S)
fi


CWD=$(pwd) #current working directory
OUT_FOLDER=$CWD"/output/"${NAME}"-FARM-"${UNIQUE_TAG}"/"

mkdir -p $CWD"/output/" #just in case output folder is not made
mkdir $OUT_FOLDER #Now make folder where all the output will go


IN_FILE=$CWD"/"$FILENAME
LOG_FILE=$OUT_FOLDER$NAME".log"
cp $IN_FILE $OUT_FOLDER

s=$OUT_FOLDER$NAME"_SLURM.txt"
#i=$SLURM_ARRAY_TASK_ID






if [[ $2 == "farm" ]]; then    #                                                          Creates a variable in lammps ${output_folder}
    srun /home/agoga/sandbox/lammps/lmp_mpi -nocite -log $LOG_FILE -in $OUT_FOLDER$FILENAME -var output_folder $OUT_FOLDER
else
    mpirun -np 2 lmp_mpi -nocite -log $LOG_FILE -in $OUT_FOLDER$FILENAME -var output_folder $OUT_FOLDER
fi
