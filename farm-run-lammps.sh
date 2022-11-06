#!/bin/bash -l
#! -cwd
#! -j y
#! -S /bin/bash
#SBATCH -D ./
#SBATCH --job-name=a-SiO2ws
#SBATCH --partition=high2 # Partition you are running on. Options: low2, med2, high2
#SBATCH --output=/home/agoga/sandbox/lammps/topcon/slurm-output/SiO-%j.txt
#SBATCH --mail-user="adgoga@ucdavis.edu"
#SBATCH --mail-type=FAIL,END


#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1 
#SBATCH --mem=64G
#SBATCH -t 4-00:00


lmppre='lmp/'
FILE=$1 #first input should be like "lmp/SilicaAnneal.lmp"
FILENAME=${1#"$lmppre"} 

j=$SLURM_JOB_ID

NAME=${FILENAME%.*}

UNIQUE_TAG="-FARM-"$SLURM_JOBID

CWD=$(pwd) #current working directory
OUT_FOLDER=$CWD"/output/"${NAME}${UNIQUE_TAG}"/"


mkdir -p $CWD"/output/" #just in case output folder is not made
mkdir $OUT_FOLDER #Now make folder where all the output will go


#copy the file to run into the new directory
IN_FILE=$CWD"/"$FILE
LOG_FILE=$OUT_FOLDER$NAME".log"
cp $IN_FILE $OUT_FOLDER

s=$OUT_FOLDER$NAME"_SLURM.txt"



export OMP_NUM_THREADS=1
# export OMP_PLACES=threads
# export OMP_PROC_BIND=true

#                                                          Creates a variable in lammps ${output_folder}
if srun /home/agoga/sandbox/lammps/lmp_mpi -nocite -log $LOG_FILE -in $OUT_FOLDER$FILENAME -var output_folder $OUT_FOLDER ; 
then #rename the output directory to show it's finished
    #srun completed succesfully 
    mv $OUT_FOLDER ${OUT_FOLDER%?}"-S"
else
    #srun failed somehow
    mv $OUT_FOLDER ${OUT_FOLDER%?}"-F"
    exit 1
fi


