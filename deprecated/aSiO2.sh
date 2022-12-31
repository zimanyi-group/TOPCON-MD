#!/bin/bash -l
#! -cwd
#! -j y
#! -S /bin/bash
#SBATCH -D ./
#SBATCH --job-name=a-SiO2
#SBATCH --partition=high2 # Partition you are running on. Options: low2, med2, high2
#SBATCH --output=/home/agoga/sandbox/lammps/topcon/slurm-output/SiO-%j.txt
#SBATCH --mail-user="adgoga@ucdavis.edu"
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-type=FAIL


#======
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=32 
#SBATCH --cpus-per-task=1 
#SBATCH --mem=64G
#SBATCH -t 10-00:00



#blahhh SBATCH --array=0-1

FILENAME="SilicaAnneal.lmp"

export OMP_NUM_THREADS=1
NAME=${FILENAME%.*}
UNIQUE_TAG=$(date +%m%d-%H%M%S)
CWD=$(pwd) #current working directory
OUT_FOLDER=$CWD"/output/"${NAME}${UNIQUE_TAG}"/"
mkdir -p $CWD"/output/" #just in case output folder is not made
mkdir $OUT_FOLDER #Now make folder where all the output will go


IN_FILE=$CWD"/"$FILENAME
LOG_FILE=$OUT_FOLDER$NAME".log"
cp $IN_FILE $OUT_FOLDER

s=$OUT_FOLDER$NAME"_SLURM.txt"
#i=$SLURM_ARRAY_TASK_ID

# echo $LOG_FILE 
# echo $IN_FILE
# echo $OUT_FOLDER



j=$SLURM_JOB_ID
PART=16

if [[ $1 == "farm" ]];
then #                                                                  Creates a variable in lammps ${output_folder}
srun /home/agoga/sandbox/lammps/lmp_mpi -nocite -log $LOG_FILE -in $IN_FILE -var output_folder $OUT_FOLDER
else
mpirun -np 2 lmp_mpi -nocite -log $LOG_FILE -in $IN_FILE -var output_folder $OUT_FOLDER
fi