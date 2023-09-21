#!/bin/bash
#SBATCH --job-name=tcNEB
#SBATCH --partition=med2
#SBATCH --output=/home/agoga/sandbox/topcon/slurm-output/j-%j.txt
#SBATCH --mail-user="adgoga@ucdavis.edu"
#SBATCH --mail-type=FAIL,END

#SBATCH --ntasks=247
#SBATCH --ntasks-per-node=247
#SBATCH --cpus-per-task=1 
#SBATCH --mem=256G
#SBATCH -t 4-0


j=$SLURM_JOB_ID

neb_file_name=NEB.lmp
neb_file=/home/agoga/sandbox/topcon/lmp/$neb_file_name

export OMP_NUM_THREADS=1

etol=0.01
timestep=1


num_replica=13
skippes=1
numruns=0
start=`date +%s`
maxneb=3000
maxclimb=1000
springconst=1
plot=true
create_gif=false

#data_folder=/home/agoga/documents/code/topcon-md/data/neb #/pinhole-dump-files/"


base_folder="/home/agoga/sandbox/topcon/" 
#datafile=$DATA_FOLDER

data_file=$base_folder$1
setName=$2 #perpPairs/
distFolder=$base_folder"data/neb/"$setName

I=${1##*/}
echo $I
samplename=${I%.*}
echo $samplename
echo $distFolder


nebfolder="/home/agoga/sandbox/topcon/neb/"$setName$j"-"$samplename"/"
mkdir -v -p $nebfolder

pairsfile=${data_file%.*}"-pairlist.txt"
echo $pairsfile
 

#styles of NEB avail
single_zap="single_zap"
multi_zap="multi_zap"
multi_jump="multi_jump"
single_jump="single_jump"
boomerang="boomerang"


########################################################
########################################################
style=$boomerang
########################################################
########################################################

cyclelen=1

mapfile -t pairs < $pairsfile

for pair in "${pairs[@]}" # #"5976 5979" #
do

    pairarray=($pair)

    #Single zap
    if [[ $style == $single_zap ]];then

        atom_id=${pairarray[0]} 
        zap_id=${pairarray[1]}

        run_id="$atom_id-$zap_id"
        echo_string="to zap from "$run_id
        create_gif=false
        

    #Multi zap(OH migration)
    elif [[ $style == $multi_zap ]];then
        run_id="$atom_id"
        echo_string="to multizap "$run_id
        #@TODO fill out


    elif [[ $style == $multi_jump ]];then
        

        atom_id=790
        j1="791 766"
        j2="766 1197"
        j3="1197 1196"
        j0="786 791"
        cyclelen=3
        jumpPairsBase="$j1,$j2,$j3" #,$j2,$j1,$j0"
        jumpPairs=$jumpPairsBase
        for ((i=1;i<1;i+=1))
        do
            jumpPairs="$jumpPairs,$jumpPairsBase"
        done

        run_id="$atom_id"
        echo_string="to multijump "$run_id
        create_gif=true

    elif [[ $style == $single_jump ]];then
        atom_id=${pairarray[0]} 
        fPosx=${pairarray[1]}
        fPosy=${pairarray[2]}
        fPosz=${pairarray[3]}
        
        atomF1=${pairarray[4]}
        atomF2=${pairarray[5]}

        run_id="$atom_id"
        echo_string="to jump "$run_id" to BC of "$atomF1"-"$atomF2
        create_gif=true 

    elif [[ $style == $boomerang ]];then
        cyclelen=2
        atom_id=${pairarray[0]} 
        fPosx=${pairarray[1]}
        fPosy=${pairarray[2]}
        fPosz=${pairarray[3]}
        
        atomF1=${pairarray[4]}
        atomF2=${pairarray[5]}
        num_repeat=100

        run_id="$atom_id"
        echo_string="to jump "$run_id" to BC of "$atomF1"-"$atomF2
        create_gif=true 
        

    fi
    

    for etol in 7e-6 #7e-6  #3e-7 1e-7 # 1e-5 #3e-6 1e-6 7e-7 5e-7 3e-7 1e-7  #7e-5 5e-5 3e-5 1e-5 
    do 
        for timestep in 0.5 #$(seq 1.6 0.1 1.8) 
        do
            
            numruns=$((numruns+1))


            unique_tag=$run_id"_"$(date +%H%M%S)

            cwd=$(pwd) #current working directory
            out_folder="/scratch/agoga/output/neb"${unique_tag}"/"
            #out_folder=$cwd"/output/neb"${unique_tag}"/" #for testing
            log_folder=$out_folder"/logs/"

            # mkdir -p $cwd"/output/" #just in case output folder is not made
            # mkdir $out_folder #Now make folder where all the output will go
            mkdir -p $log_folder

            
            #log_file=$log_folder$atomnum"neb.log"
            
            cp /home/agoga/sandbox/topcon/py/PrepNEB.py $out_folder
            cp /home/agoga/sandbox/topcon/py/Process-NEB.py $out_folder 
            cp $neb_file $out_folder

            s=$out_folder$NAME"_SLURM.txt"

            neb_info_file=$out_folder"nebinfo.txt"
            
            

            echo "----------------Prepping NEB "$echo_string" ----------------"
            srun /home/agoga/anaconda3/envs/lmp/bin/python $out_folder"PrepNEB.py" \
            --out=$out_folder --etol=$etol --ts=$timestep --dfile=$data_file --plot=$plot --atomid=$atom_id --info=$neb_info_file \
            --style=$style --fposx=$fPosx --fposy=$fPosy --fposz=$fPosz --bc1=$atomF1 --bc2=$atomF2 --bclist="$jumpPairs" --repeat=$num_repeat
            
            
            while read -u3 line
            do
                #echo $line
                
                if [[ $line == neb* ]] #if the line starts with "neb"
                then
                    nebline=($line)
                    neb_atom_id=${nebline[2]}
                    neb_identifier=${nebline[3]}
                    nebI=${nebline[4]}
                    nebF=${nebline[5]}
                    logf=${nebline[6]}
                    h_id=${nebline[7]}
                    log_file=$log_folder$logf

                    echo "----------------Running NEB for "$neb_identifier" ----------------"
                    srun /home/agoga/.local/bin/lmp_mpi -partition "$num_replica"x19 \
                        -nocite -log $log_file -in $out_folder$neb_file_name -var maxneb $maxneb -var maxclimb $maxclimb \
                        -var output_folder $out_folder -var ts $timestep -var etol $etol  -var springconst $springconst -pscreen none -screen none \
                        -var identifier $neb_identifier -var h_id $h_id -var atom_id $neb_atom_id -var nebI $nebI -var nebF $nebF    
                        # $out_folder/screen         
                fi

                
            done 3< "$neb_info_file"

            echo "----------------Post NEB "$echo_string"  ----------------"
            srun /home/agoga/anaconda3/envs/lmp/bin/python $out_folder"Process-NEB.py" \
            --out=$out_folder --etol=$etol --ts=$timestep --nebfolder=$nebfolder --dfile=$data_file \
            --k=$springconst --plot=$plot --info=$neb_info_file --style=$style --gif=$create_gif --neblog=$log_file \
            --atomid=$atom_id --cylen=$cyclelen 
        
        done
    done
done

end=`date +%s`

runtime=$( echo "$end-$start" | bc -l)
runtimeMin=$( echo "$runtime/60" | bc -l)
runtimeAvg=$( echo "$runtimeMin/$numruns" | bc -l)
echo "Total runtime:" $runtimeMin"m"
echo "AVG run time:" $runtimeAvg"m"
