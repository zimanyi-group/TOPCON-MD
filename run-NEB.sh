#!/bin/bash -l
#! -cwd
#! -j y
#! -S /bin/bash


lmppre="lmp/"
FILE="lmp/NEB.lmp"
FILENAME=${FILE#"$lmppre"}


export OMP_NUM_THREADS=1

ETOL=0.01
TIMESTEP=1


SKIPPES=1
numruns=0
start=`date +%s`
MAXNEB=3000
MAXCLIMB=1000
SPRINGCONST=1
plot=0
DATA_FOLDER=/home/agoga/documents/code/topcon-md/data/neb #/pinhole-dump-files/"

setName=/FullSet
distFolder=$DATA_FOLDER$setName


for DATAFILE in "$distFolder"/*.dat
do
    
    # DATAFILE=$DATA_FOLDER"SiOxNEB-NOH.data"
    # # DATAFILE="SiOxNEB-NOH.data"
    PAIRSFILE=${DATAFILE%.*}"-pairlist.txt"
    echo $PAIRSFILE
    NEBFOLDER="/home/agoga/documents/code/topcon-md/neb-out/"
    mkdir -p $NEBFOLDER$setName

    lastdone="385 1189"
    alreadydone=1 #set to 1 to run everything



    mapfile -t pairs < $PAIRSFILE

    # mapfile -t pairs < data/Hpairs-v1.txt
    # ATOMNUM=1642 #2041 #3898 #3339 #H #4090 #1649 #2776 #1659 
    # ATOMREMOVE=1638
    for pair in "${pairs[@]}" # #"5976 5979" #
    do
        if [[ $alreadydone  == 0  && $pair != $lastdone ]] ; then
            continue
        else
            alreadydone=1
        fi

        ATOMNUM=${pair% *}
        ATOMREMOVE=${pair#* }


        for ETOL in 7e-6 #5e-6 #3e-7 1e-7 # 1e-5 #3e-6 1e-6 7e-7 5e-7 3e-7 1e-7  #7e-5 5e-5 3e-5 1e-5 
        do 
            for TIMESTEP in 0.5 #$(seq 1.6 0.1 1.8) 
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
            


                echo "----------------Prepping NEB for "$pair" ----------------"
                mpirun -np 4 python3 /home/agoga/documents/code/topcon-md/py/FindMinimumE.py $OUT_FOLDER $ATOMNUM $ETOL $TIMESTEP $SKIPPES $ATOMREMOVE $DATAFILE $plot
                
                echo "----------------Running NEB for "$pair" ----------------"
                mpirun -np 13 --oversubscribe lmp_mpi -partition 13x1 -nocite -log $LOG_FILE -in $OUT_FOLDER$FILENAME -var maxneb $MAXNEB -var maxclimb $MAXCLIMB -var atom_id $ATOMNUM -var output_folder $OUT_FOLDER -var fileID $ATOMNUM -var etol $ETOL -var ts $TIMESTEP -var springconst $SPRINGCONST -pscreen $OUT_FOLDER/screen
                
                echo "----------------Post NEB for "$pair" ----------------"
                python3 /home/agoga/documents/code/topcon-md/py/Process-NEB.py $OUT_FOLDER $ATOMNUM $ETOL $TIMESTEP $ATOMREMOVE $NEBFOLDER $DATAFILE $SPRINGCONST $plot
            
            done
        done
    done

    end=`date +%s`

    runtime=$( echo "$end-$start" | bc -l)
    runtimeMin=$( echo "$runtime/60" | bc -l)
    runtimeAvg=$( echo "$runtimeMin/$numruns" | bc -l)
    echo "Total runtime:" $runtimeMin"m"
    echo "AVG run time:" $runtimeAvg"m"
done