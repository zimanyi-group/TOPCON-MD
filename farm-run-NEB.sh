#!/bin/sh
#SBATCH --job-name=nebtst
#SBATCH --partition=med2
#SBATCH --output=/home/agoga/sandbox/topcon/slurm-output/j-%j.txt
#SBATCH --mail-user="adgoga@ucdavis.edu"
#SBATCH --mail-type=FAIL,END

#SBATCH --ntasks=247
#SBATCH --ntasks-per-node=256
#SBATCH --cpus-per-task=1 
#SBATCH --mem=0
#SBATCH -t 4-0



lmppre='lmp/'
j=$SLURM_JOB_ID

export OMP_NUM_THREADS=1


ETOL=0.01
TIMESTEP=1
SKIPPES=1
numruns=0
start=`date +%s`

MAXNEB=3000
MAXCLIMB=1000

NEBFOLDER="/home/agoga/documents/code/topcon-md/output/NEB"
#mapfile -t pairs < data/noHpairs-v1.txt

# for pair in "${pairs[@]}"

for pair in "1378 1382" #"${pairs[@]}" #"3014 3012" #
do
    ATOMNUM=${pair% *}
    ATOMREMOVE=${pair#* }
    for ETOL in 1e-7 #1e-6 1e-7 #3e-5 7e-5 7e-6 5e-6 3e-6 1e-6 7e-7 5e-7 3e-7 1e-7
    do 
        for TIMESTEP in 0.5 #$(seq 0.1 0.1 0.9) 
        do
        # for ATOMREMOVE in 3090 #4929 #3715 # 3341 # 3880  #1548 1545 3955 3632 3599
        # do
            
            numruns=$((numruns+1))
            # NAME=${FILENAME%.*}


            UNIQUE_TAG=$ATOMNUM"-"$ATOMREMOVE"_"$TIMESTEP"-"$ETOL"_"$(date +%H%M%S)

            CWD=$(pwd) #current working directory
            #OUT_FOLDER=$CWD"/output/"${NAME}${UNIQUE_TAG}"/"
            OUT_FOLDER=$CWD"/output/neb-"${UNIQUE_TAG}"/"
            DATA_FOLDER=$OUT_FOLDER"/logs/"

            mkdir -p $CWD"/output/" #just in case output folder is not made
            mkdir $OUT_FOLDER #Now make folder where all the output will go
            mkdir $DATA_FOLDER


            IN_FILE=$CWD"/"$FILE

            LOG_FILE=$DATA_FOLDER$ATOMNUM"neb.log"
            NEB_FILE=/home/agoga/sandbox/topcon/lmp/NEB.lmp
            cp /home/agoga/sandbox/topcon/py/FindMinimumE.py $OUT_FOLDER
            cp $NEB_FILE $OUT_FOLDER

            s=$OUT_FOLDER$NAME"_SLURM.txt"


            echo "----------------Prepping NEB----------------"
            srun /home/agoga/.conda/envs/lmp/bin/python /home/agoga/sandbox/topcon/py/FindMinimumE.py $OUT_FOLDER $ATOMNUM $ETOL $TIMESTEP $SKIPPES $ATOMREMOVE
            
            echo "----------------Running NEB----------------"
            srun /home/agoga/lammps-23Jun2022/build/lmp_mpi -partition 13x19 -nocite -log $LOG_FILE -in $NEB_FILE -var maxneb ${MAXNEB} -var maxclimb ${MAXCLIMB} -var atom_id ${ATOMNUM} -var output_folder $OUT_FOLDER -var fileID $ATOMNUM -var etol $ETOL -var ts $TIMESTEP -pscreen $OUT_FOLDER/screen
            
            echo "----------------Post NEB----------------"
            srun /home/agoga/.conda/envs/lmp/bin/python /home/agoga/sandbox/topcon/py/Process-NEB.py $OUT_FOLDER $ATOMNUM $ETOL $TIMESTEP $ATOMREMOVE $NEBFOLDER
        
        done
    done
done

end=`date +%s`

runtime=$( echo "$end-$start" | bc -l)
runtimeMin=$( echo "$runtime/60" | bc -l)
runtimeAvg=$( echo "$runtimeMin/$numruns" | bc -l)
echo "Total runtime:" $runtimeMin"m"
echo "AVG run time:" $runtimeAvg"m"