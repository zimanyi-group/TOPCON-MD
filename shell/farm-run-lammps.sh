#!/bin/bash -l
#! -cwd
#! -j y
#! -S /bin/bash

#Serialize the input first
#Inputs
#1-filename string
#2-job-name
#3-cpus
#4-priority
#5-time limit
usagestr='Usage $0 filename jobname cpus [priority] [timelimit]'
#echo $usagestr
CWD=$(pwd)
lmppre='lmp/'
#if no filename jobname or cpus
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ];
then
    echo $usagestr
    exit 1
fi

#priority
if [ -z "$4" ];
then
    set -- "$@" 'med2' #modifies the command line argument 4  - https://stackoverflow.com/questions/61096448/how-to-modify-command-line-arguments-inside-bash-script-using-set
fi

#timelimit
if [ -z "$5" ]; #if no time given then default to 4 days
then
    set -- "$@" '4-0'
elif ["$5" != *"-"*];
then
    echo $usagestr
    echo 'Acceptable time formats include "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds", maybe I should fix this if I want shorter times.'
    exit 1
fi



sbatch <<-EOT
#!/bin/sh
#SBATCH -D ./
#SBATCH --job-name=$2
#SBATCH --partition=$4 # Partition you are running on. Options: low2, med2, high2
#SBATCH --output=/home/agoga/sandbox/topcon/slurm-output/j-%j.txt
#SBATCH --mail-user="adgoga@ucdavis.edu"
#SBATCH --mail-type=FAIL,EN


#SBATCH --ntasks=$3
#SBATCH --ntasks-per-node=$3
#SBATCH --cpus-per-task=1 
#SBATCH --mem=250G
#SBATCH -t $5

j=\$SLURM_JOB_ID

CWD=$(pwd) #current working directory



FILE=$1 #first input should be like "lmp/SilicaAnneal.lmp"
FILENAME=${1#"$lmppre"} #remove lmp/ from file if there

NAME=\${FILENAME%.*}

UNIQUE_TAG="-$2-"\$SLURM_JOBID

OUT_FOLDER=$CWD"/output/"\${NAME}\${UNIQUE_TAG}"/"
mkdir -p $CWD"/output/" #just in case output folder is not made

mkdir \$OUT_FOLDER #Now make folder where all the output will go



#copy the file to run into the new directory
IN_FILE=$CWD"/"\$FILE
LOG_FILE=\$OUT_FOLDER\$NAME".log"
cp \$IN_FILE \$OUT_FOLDER




export OMP_NUM_THREADS=2
# export OMP_PLACES=threads
# export OMP_PROC_BIND=true

SF=/home/agoga/sandbox/topcon/slurm-output/j-\$j.txt

#                       Creates a variable in lammps \${output_folder}
if srun /home/agoga/lammps-29Aug2024/build/lmp_mpi -nocite -log \$LOG_FILE -in \$OUT_FOLDER\$FILENAME -var output_folder \$OUT_FOLDER ; 

#after srun exits
then #rename the output directory to show it's finished
    #srun completed succesfully 
    mv \$OUT_FOLDER \${OUT_FOLDER%?}"-S"
    S=\${OUT_FOLDER%?}"-S"/j-\$j".txt"
    mv \$SF \$S
    exit 0
else
    #srun failed somehow
    mv \$OUT_FOLDER \${OUT_FOLDER%?}"-F"
    S=\${OUT_FOLDER%?}"-F"/j-\$j".txt"
    mv \$SF \$S
    exit 1
fi
EOT