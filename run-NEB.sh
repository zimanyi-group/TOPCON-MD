#!/bin/bash -l
#! -cwd
#! -j y
#! -S /bin/bash

export OMP_NUM_THREADS=1

###COMMAND LINE ARGUMENTS
###1ST FILE NAME

lmppre='lmp/'
FILE=$1
FILENAME=${1#"$lmppre"}

j=$SLURM_JOB_ID


ETOL=0.01
TIMESTEP=1
SKIPPES=1
numruns=0
start=`date +%s`

mapfile -t pairs < data/Hpairs-v1.txt
# ATOMNUM=1642 #2041 #3898 #3339 #H #4090 #1649 #2776 #1659 
# ATOMREMOVE=1638
for pair in "${pairs[@]}"
do

    ATOMNUM=${pair% *}
    ATOMREMOVE=${pair#* }

    for ETOL in 1e-5 7e-6 5e-6 #1e-7 7e-8 5e-8 3e-8 1e-8 #7e-5 5e-5 3e-5 1e-5 7e-6 5e-6 3e-6 1e-6 7e-7 5e-7 3e-7 #
    do 
        for TIMESTEP in 0.5 #$(seq 0.2 0.05 0.85) 
        do
        # for ATOMREMOVE in 3090 #4929 #3715 # 3341 # 3880  #1548 1545 3955 3632 3599
        # do
            
            ((numruns++))
            NAME=${FILENAME%.*}


            UNIQUE_TAG=$ATOMNUM"-"$ATOMREMOVE"_"$TIMESTEP"-"$ETOL"_"$(date +%H%M%S)

            CWD=$(pwd) #current working directory
            OUT_FOLDER=$CWD"/output/"${NAME}${UNIQUE_TAG}"/"
            DATA_FOLDER=$OUT_FOLDER"/logs/"

            mkdir -p $CWD"/output/" #just in case output folder is not made
            mkdir $OUT_FOLDER #Now make folder where all the output will go
            mkdir $DATA_FOLDER


            IN_FILE=$CWD"/"$FILE

            LOG_FILE=$DATA_FOLDER$ATOMNUM"neb.log"
            cp $IN_FILE $OUT_FOLDER


            s=$OUT_FOLDER$NAME"_SLURM.txt"
        


            echo "----------------Prepping NEB----------------"
            mpirun -np 4 python3 /home/agoga/documents/code/topcon-md/py/FindMinimumE.py $OUT_FOLDER $ATOMNUM $ETOL $TIMESTEP $SKIPPES $ATOMREMOVE
            
            echo "----------------Running NEB----------------"
            mpirun -np 13 --oversubscribe lmp_mpi -partition 13x1 -nocite -log $LOG_FILE -in $OUT_FOLDER$FILENAME -var atom_id ${ATOMNUM} -var output_folder $OUT_FOLDER -var fileID $ATOMNUM -var etol $ETOL -var ts $TIMESTEP -pscreen $OUT_FOLDER/screen
            
            echo "----------------Post NEB----------------"
            python3 /home/agoga/documents/code/topcon-md/py/Process-NEB.py $OUT_FOLDER $ATOMNUM $ETOL $TIMESTEP $ATOMREMOVE
        
        done
    done
done

end=`date +%s`

runtime=$( echo "$end-$start" | bc -l)
runtimeMin=$( echo "$runtime/60" | bc -l)
runtimeAvg=$( echo "$runtimeMin/$numruns" | bc -l)
echo "Total runtime:" $runtimeMin"m"
echo "AVG run time:" $runtimeAvg"m"