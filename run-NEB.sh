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


REPLICAS=9


ATOMNUM=125 #4090 #1649 #2776 #1659 

#126,344,339
#134,124,133
ATOMREMOVE=133

#mirror pairs
#126 133 
#344 134
#339 124

ETOL=0.01
TIMESTEP=1
SKIPPES=1

for TIMESTEP in 0.5 #0.8 0.9 1.0 #0.1 0.2 0.3 0.5 0.4 0.6 0.7  #1.2 1.4 1.1 1.3 1.5 1.6 1.7 1.8 1.9 2.0 #0.01 0.05 0.07 0.03
do
    #  
    for ETOL in 0.000000001 # 0.00000005 0.000000005 #0.0001 0.00001 0.00005 0.000001 0.000005 0.0000001 0.0000005 # #0.01 0.05 .005 0.001 0.0005 
    do
    #134 124 133
        for ATOMREMOVE in 133 #134 124 133 #126 344 339 
        do
            NAME=${FILENAME%.*}


            UNIQUE_TAG=$ATOMNUM"-"$ATOMREMOVE"_"$TIMESTEP"-"$ETOL"_"$(date +%H%M%S)



            CWD=$(pwd) #current working directory
            OUT_FOLDER=$CWD"/output/"${NAME}${UNIQUE_TAG}"/"
            DATA_FOLDER=$OUT_FOLDER"/logs/"

            mkdir -p $CWD"/output/" #just in case output folder is not made
            mkdir $OUT_FOLDER #Now make folder where all the output will go
            mkdir $DATA_FOLDER


            IN_FILE=$CWD"/"$FILE
            #LOG_FILE=$OUT_FOLDER$NAME".log"
            LOG_FILE=$DATA_FOLDER$ATOMNUM"neb.log"
            cp $IN_FILE $OUT_FOLDER


            s=$OUT_FOLDER$NAME"_SLURM.txt"


            #export OMP_NUM_THREADS=1



            python3 /home/agoga/documents/code/topcon-md/py/FindMinimumE.py $OUT_FOLDER $ATOMNUM $ETOL $TIMESTEP $SKIPPES $ATOMREMOVE

            mpirun -np 11 --oversubscribe lmp_mpi -partition 11x1 -nocite -log $LOG_FILE -in $OUT_FOLDER$FILENAME -var atom_id ${ATOMNUM} -var output_folder $OUT_FOLDER -var fileID $ATOMNUM -var etol $ETOL -var ts $TIMESTEP -pscreen $OUT_FOLDER/screen

            python3 /home/agoga/documents/code/topcon-md/py/Process-NEB.py $OUT_FOLDER $ATOMNUM $ETOL $TIMESTEP $ATOMREMOVE
        done
    done
done