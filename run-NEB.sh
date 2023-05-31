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


ATOMNUM=3898 #3924 #4090 #1649 #2776 #1659 

#4624 4030


ETOL=0.01
TIMESTEP=1
SKIPPES=1
numruns=0
start=`date +%s`

for ETOL in  1e-5 7e-6 5e-6 # 3e-6 1e-6 #1e-4 1e-5 5e-5 1e-6 5e-6 # #5e-9 1e-9 #5e-5 1e-5 5e-6 #5e-4 1e-5 5e-5 1e-6 5e-6 
do 
    for TIMESTEP in 0.4 #$(seq 0.2 0.05 0.6) #0.1 0.2 0.3 0.4 #1.2 1.4 1.1 1.3 1.5 1.6 1.7 1.8 1.9 2.0 #0.01 0.05 0.07 0.03
    do
    #134 124 133
        for ATOMREMOVE in 3715 #3880 #1547 #1548 1545 3955 3632 3599
        do
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
            #LOG_FILE=$OUT_FOLDER$NAME".log"
            LOG_FILE=$DATA_FOLDER$ATOMNUM"neb.log"
            cp $IN_FILE $OUT_FOLDER


            s=$OUT_FOLDER$NAME"_SLURM.txt"


            #export OMP_NUM_THREADS=1


            echo "----------------Prepping NEB----------------"
            python3 /home/agoga/documents/code/topcon-md/py/FindMinimumE.py $OUT_FOLDER $ATOMNUM $ETOL $TIMESTEP $SKIPPES $ATOMREMOVE
            echo "----------------Running NEB----------------"
            mpirun -np 11 --oversubscribe lmp_mpi -partition 11x1 -nocite -log $LOG_FILE -in $OUT_FOLDER$FILENAME -var atom_id ${ATOMNUM} -var output_folder $OUT_FOLDER -var fileID $ATOMNUM -var etol $ETOL -var ts $TIMESTEP -pscreen $OUT_FOLDER/screen
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