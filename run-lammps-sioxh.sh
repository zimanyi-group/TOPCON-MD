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
LOG_FILE=$OUT_FOLDER$NAME".log"

cp $IN_FILE $OUT_FOLDER

s=$OUT_FOLDER$NAME"_SLURM.txt"




SEED=12345
NUM_SI=24
NUM_O=40
NUM_H=2

export OMP_NUM_THREADS=1
# export OMP_PLACES=threads
# export OMP_PROC_BIND=true

for dump in $(find /home/agoga/documents/code/topcon-md/data/zeke/new -name "*.lammps" -type f -print)
do
    for NUM_H in  {1..12}
    do
        mpirun -np 4 lmp_mpi -nocite -log $LOG_FILE -in $OUT_FOLDER$FILENAME -var output_folder $OUT_FOLDER -var dump_file $dump -var seed $SEED -var numSi $NUM_SI -var numO $NUM_O -var numH $NUM_H
    done
done

# for NUM_O in 30 #36 40 #{24..48}
# do
#     for NUM_H in  1 #{1..12}
#     do
#         SEED=$RANDOM #$((NUM_O*100 + NUM_H))
#         mpirun -np 4 lmp_mpi -nocite -log $LOG_FILE -in $OUT_FOLDER$FILENAME -var output_folder $OUT_FOLDER -var seed $SEED -var numSi $NUM_SI -var numO $NUM_O -var numH $NUM_H
#     done
# done