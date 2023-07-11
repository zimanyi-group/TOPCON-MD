#!/bin/bash -l
#! -cwd
#! -j y
#! -S /bin/bash


###COMMAND LINE ARGUMENTS
###1ST FILE NAME



lmppre='lmp/'
FILE=$1
FILENAME=${1#"$lmppre"}

j=$SLURM_JOB_ID

NAME=${FILENAME%.*}


UNIQUE_TAG=$(date +%m%d-%Hh%Mm%S)



CWD=$(pwd) #current working directory
OUT_FOLDER=$CWD"/output/"${NAME}${UNIQUE_TAG}"/"

mkdir -p $CWD"/output/" #just in case output folder is not made
mkdir $OUT_FOLDER #Now make folder where all the output will go


IN_FILE=$CWD"/"$FILE
#LOG_FILE=$OUT_FOLDER$NAME".log"
LOG_FILE="/home/agoga/documents/code/topcon-md/data/HNEB1/2661-05.log"
cp $IN_FILE $OUT_FOLDER

s=$OUT_FOLDER$NAME"_SLURM.txt"






export OMP_NUM_THREADS=1
# export OMP_PLACES=threads
# export OMP_PROC_BIND=true


mpirun -np 4 lmp_mpi -nocite -log $LOG_FILE -in $OUT_FOLDER$FILENAME -var output_folder $OUT_FOLDER
#lmp_mpi -partition 1x1 -nocite -log $LOG_FILE -in $OUT_FOLDER$FILENAME -var output_folder $OUT_FOLDER
