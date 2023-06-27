#!/bin/sh
#SBATCH -D ./
#SBATCH --job-name=nebtest
#SBATCH --partition=high2 # Partition you are running on. Options: low2, med2, high2
#SBATCH --output=/home/agoga/sandbox/topcon/slurm-output/j-%j.txt
#SBATCH --mail-user="adgoga@ucdavis.edu"
#SBATCH --mail-type=FAIL,END


#SBATCH --ntasks=256
#SBATCH --ntasks-per-node=256
#SBATCH --cpus-per-task=1 
#SBATCH --mem=64G
#SBATCH -t 4-0

j=\$SLURM_JOB_ID

lmppre='lmp/'
FILE=$1
FILENAME=${1#"$lmppre"}

j=$SLURM_JOB_ID



ATOMNUM=1642 #2041 #3898 #3339 #H #4090 #1649 #2776 #1659 

#4624 4030

ETOL=0.01
TIMESTEP=1
SKIPPES=1
numruns=0
start=`date +%s`

#mapfile -t pairs < data/noHpairs-v1.txt

# for pair in "${pairs[@]}"
# do

#     ATOMNUM=${pair% *}
#     ATOMREMOVE=${pair#* }
ATOMREMOVE=1638
for ETOL in 1e-5 #1e-6 1e-7 #3e-5 7e-5 7e-6 5e-6 3e-6 1e-6 7e-7 5e-7 3e-7 1e-7
do 
    for TIMESTEP in 0.5 #$(seq 0.1 0.1 0.9) 
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


        #export OMP_NUM_THREADS=1


        echo "----------------Prepping NEB----------------"
        srun /share/apps/conda3/miniconda3/bin/python3 //home/agoga/sandbox/topcon/py/FindMinimumE.py $OUT_FOLDER $ATOMNUM $ETOL $TIMESTEP $SKIPPES $ATOMREMOVE
        echo "----------------Running NEB----------------"
        
        #mpirun -np 13 --oversubscribe lmp_mpi -partition 13x1 -nocite -log $LOG_FILE -in $OUT_FOLDER$FILENAME -var atom_id ${ATOMNUM} -var output_folder $OUT_FOLDER -var fileID $ATOMNUM -var etol $ETOL -var ts $TIMESTEP -pscreen $OUT_FOLDER/screen
        echo "----------------Post NEB----------------"
        #srun /share/apps/conda3/miniconda3/bin/python3 /home/agoga/sandbox/topcon/py/Process-NEB.py $OUT_FOLDER $ATOMNUM $ETOL $TIMESTEP $ATOMREMOVE
    
    done
done
#done

end=`date +%s`

runtime=$( echo "$end-$start" | bc -l)
runtimeMin=$( echo "$runtime/60" | bc -l)
runtimeAvg=$( echo "$runtimeMin/$numruns" | bc -l)
echo "Total runtime:" $runtimeMin"m"
echo "AVG run time:" $runtimeAvg"m"