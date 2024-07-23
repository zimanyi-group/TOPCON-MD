#!/bin/sh
#SBATCH -D ./
#SBATCH --job-name=creDat
#SBATCH --partition=med2 # Partition you are running on. Options: low2, med2, high2
#SBATCH --output=/home/agoga/sandbox/topcon/slurm-output/j-%j.txt
#SBATCH --mail-user="adgoga@ucdavis.edu"
#SBATCH --mail-type=FAIL,END


#SBATCH --ntasks=512
#SBATCH --ntasks-per-node=256
#SBATCH --cpus-per-task=1 
#SBATCH --mem=512G
#SBATCH -t 1-0

j=$SLURM_JOB_ID

CWD=$(pwd) #current working directory

DATANAME=${1##*/}
INPUTFILE=$CWD"/"$1
echo $DATANAME
echo $INPUTFILE
# DUMPNAME=${CWD}"/"${INPUTFILE%".dump"}
# echo $DUMPNAME

#handler file
# # # ##!/bin/bash
# # # # datafolder=/home/agoga/sandbox/topcon/data/SiOxVaryH

# # # # for DATAFILE in "$datafolder"/*.dat
# # # # do
# # # #     sbatch farm-run-createdat.sh data/SiOxVaryH/${DATAFILE##*/}
# # # # done

FILENAME="CreateDat.lmp"

NAME=${FILENAME%.*}
UNIQUE_TAG="-cdat-"$SLURM_JOBID

OUT_FOLDER=$CWD"/output/"${NAME}${UNIQUE_TAG}"/"
mkdir -p $CWD"/output/" #just in case output folder is not made

mkdir $OUT_FOLDER #Now make folder where all the output will go

OUTPUTFOLDER=/home/agoga/sandbox/topcon/data/createdat/
OUTPUTFILE=$OUTPUTFOLDER$DATANAME
echo $OUTPUTFILE

#copy the file to run into the new directory
IN_FILE=$CWD"/lmp/"$FILENAME
LOG_FILE=$OUT_FOLDER$NAME".log"
cp $IN_FILE $OUT_FOLDER




export OMP_NUM_THREADS=1
# export OMP_PLACES=threads
# export OMP_PROC_BIND=true

SF=/home/agoga/sandbox/topcon/slurm-output/j-$j.txt

#                       Creates a variable in lammps ${output_folder}
if srun /home/agoga/lammps-23Jun2022/build/lmp_mpi -nocite -log $LOG_FILE -in $OUT_FOLDER$FILENAME -var infile $INPUTFILE -var outfile $OUTPUTFILE -var output_folder $OUT_FOLDER ; 

#after srun exits
then #rename the output directory to show it's finished
    #srun completed succesfully 
    mv $OUT_FOLDER ${OUT_FOLDER%?}"-S"
    S=${OUT_FOLDER%?}"-S"/j-$j".txt"
    mv $SF $S
    exit 0
else
    #srun failed somehow
    mv $OUT_FOLDER ${OUT_FOLDER%?}"-F"
    S=${OUT_FOLDER%?}"-F"/j-$j".txt"
    mv $SF $S
    exit 1
fi