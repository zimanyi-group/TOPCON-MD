#!/bin/sh
#SBATCH -D ./
#SBATCH --job-name=crePair
#SBATCH --partition=high2 # Partition you are running on. Options: low2, med2, high2
#SBATCH --output=/home/agoga/sandbox/topcon/slurm-output/j-%j.txt
#SBATCH --mail-user="adgoga@ucdavis.edu"
#SBATCH --mail-type=FAIL,END


#SBATCH --ntasks=21
#SBATCH --ntasks-per-node=21
#SBATCH --cpus-per-task=1 
#SBATCH --mem=128G
#SBATCH -t 4-0

j=$SLURM_JOB_ID

CWD=$(pwd) #current working directory

# DATANAME=${1##*/}
# INPUTFILE=$CWD"/"$1
# echo $DATANAME
# echo $INPUTFILE
# DUMPNAME=${CWD}"/"${INPUTFILE%".dump"}
# echo $DUMPNAME

#handler file
# # # ##!/bin/bash
# # # # datafolder=/home/agoga/sandbox/topcon/data/SiOxVaryH

# # # # for DATAFILE in "$datafolder"/*.dat
# # # # do
# # # #     sbatch farm-run-createdat.sh data/SiOxVaryH/${DATAFILE##*/}
# # # # done

FILENAME="CreatePairList.py"
supportFile="NEBTools.py"

NAME=${FILENAME%.*}
UNIQUE_TAG="-cpair-"$SLURM_JOBID

OUT_FOLDER=$CWD"/output/"${NAME}${UNIQUE_TAG}"/"
mkdir -p $CWD"/output/" #just in case output folder is not made

mkdir $OUT_FOLDER #Now make folder where all the output will go

# OUTPUTFOLDER=/home/agoga/sandbox/topcon/data/createpair/
# OUTPUTFILE=$OUTPUTFOLDER$DATANAME
# echo $OUTPUTFILE

#copy the file to run into the new directory
IN_FILE=$CWD"/py/"$FILENAME
sup_file=$CWD"/py/"$supportFile
LOG_FILE=$OUT_FOLDER$NAME".log"
echo $IN_FILE
cp $IN_FILE $OUT_FOLDER
cp $sup_file $OUT_FOLDER




export OMP_NUM_THREADS=1
# export OMP_PLACES=threads
# export OMP_PROC_BIND=true

SF=/home/agoga/sandbox/topcon/slurm-output/j-$j.txt

#                       Creates a variable in lammps ${output_folder}
srun /home/agoga/anaconda3/envs/lmp/bin/python $OUT_FOLDER$FILENAME
