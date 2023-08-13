#!/bin/bash
#SBATCH --job-name=tcNEB
#SBATCH --partition=med2
#SBATCH --output=/home/agoga/sandbox/topcon/slurm-output/j-%j.txt
#SBATCH --mail-user="adgoga@ucdavis.edu"
#SBATCH --mail-type=FAIL,END

#SBATCH --ntasks=13
#SBATCH --ntasks-per-node=13
#SBATCH --cpus-per-task=1 
#SBATCH --mem=64G
#SBATCH -t 4-0


#CWD=$(pwd) #current working directory
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
SPRINGCONST=1


#datafile="/home/agoga/sandbox/topcon/data/neb/Hcon-1500-110.data" #"$CWD"/"$1

# I="${1##*/}"
# ID="${I%.*}"
plot=0

base_folder="/home/agoga/sandbox/topcon/" 
#datafile=$DATA_FOLDER

datafile=$base_folder$1
setName=$2 #perpPairs/
distFolder=$base_folder"data/neb/"$setName

I=${1##*/}
echo $I
samplename=${I%.*}
echo $samplename
echo $distFolder


nebfolder="/home/agoga/sandbox/topcon/neb/"$setName$j"-"$samplename"/"
mkdir -v -p $nebfolder

pairsfile=${datafile%.*}"-pairlist.txt"
echo $pairsfile

mapfile -t pairs < $pairsfile

for pair in "${pairs[@]}" #"3014 3012" #
do
    atomnum=${pair% *}
    atomremove=${pair#* }
    for ETOL in 7e-6 # 5e-5 3e-5 1e-5 7e-6 5e-6 3e-6 1e-6 7e-7 5e-7 3e-7 1e-7 
    do 
        for TIMESTEP in 0.5 #$(seq 0.5 0.05 2) 
        do
        # for atomremove in 3090 #4929 #3715 # 3341 # 3880  #1548 1545 3955 3632 3599
        # do
            
            numruns=$((numruns+1))
            # NAME=${FILENAME%.*}


            UNIQUE_TAG=$samplename"_"$atomnum"-"$atomremove"_"$TIMESTEP"-"$ETOL"_"$(date +%H%M%S)

            CWD=$(pwd) #current working directory
            #OUT_FOLDER=$CWD"/output/neb"${UNIQUE_TAG}"/"
            OUT_FOLDER="/scratch/agoga/output/neb"${UNIQUE_TAG}"/"
            LOG_FOLDER=$OUT_FOLDER"/logs/"

            # mkdir $CWD"/output/" #just in case output folder is not made
            # mkdir $OUT_FOLDER #Now make folder where all the output will go
            mkdir -p $LOG_FOLDER


            IN_FILE=$CWD"/"$FILE

            LOG_FILE=$LOG_FOLDER$atomnum"neb.log"
            NEB_FILE=/home/agoga/sandbox/topcon/lmp/NEB.lmp
            cp /home/agoga/sandbox/topcon/py/FindMinimumE.py $OUT_FOLDER
            cp /home/agoga/sandbox/topcon/py/Process-NEB.py $OUT_FOLDER
            cp $NEB_FILE $OUT_FOLDER

            s=$OUT_FOLDER$NAME"_SLURM.txt"

#19 247 - 39 507
            echo "----------------Prepping NEB for "$pair" ----------------"
            srun /home/agoga/.conda/envs/lmp/bin/python $OUT_FOLDER"FindMinimumE.py" $OUT_FOLDER $atomnum $ETOL $TIMESTEP $SKIPPES $atomremove $datafile $plot
            
            echo "----------------Running NEB for "$pair" ----------------"
            srun /home/agoga/.local/bin/lmp_mpi -partition 13x1 -nocite -log $LOG_FILE -in $NEB_FILE -var maxneb ${MAXNEB} -var maxclimb ${MAXCLIMB} -var atom_id ${atomnum} -var output_folder $OUT_FOLDER -var fileID $atomnum -var etol $ETOL -var ts $TIMESTEP -var springconst $SPRINGCONST -pscreen $OUT_FOLDER/screen
            
            echo "----------------Post NEB for "$pair" ----------------"
            srun /home/agoga/.conda/envs/lmp/bin/python $OUT_FOLDER"Process-NEB.py" $OUT_FOLDER $atomnum $ETOL $TIMESTEP $atomremove $nebfolder $datafile $SPRINGCONST $plot

            rm -r $OUT_FOLDER
        done
    done
done


end=`date +%s`

runtime=$( echo "$end-$start" | bc -l)
runtimeMin=$( echo "$runtime/60" | bc -l)
runtimeAvg=$( echo "$runtimeMin/$numruns" | bc -l)
echo "Total runtime:" $runtimeMin"m"
echo "AVG run time:" $runtimeAvg"m"