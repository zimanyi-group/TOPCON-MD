#!/bin/bash -l
#! -cwd
#! -j y
#! -S /bin/bash



neb_file=/home/agoga/documents/code/topcon-md/lmp/NEB.lmp

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
data_folder=/home/agoga/documents/code/topcon-md/data/neb #/pinhole-dump-files/"

setName=/InsidePinhole/
setName=/PinholeCenterZap/
distFolder=$data_folder$setName"/"
nebfolder="/home/agoga/documents/code/topcon-md/neb-out/"
mkdir -p $nebfolder$setName


# for data_file in "$distFolder"/*.dat
# do
data_file="${data_folder}${setName}/pinhole_1.6-551.dat"
data_file="${data_folder}${setName}/Hcon-1500-695.dat"
#data_file="${data_folder}${setName}/boom_1.6-551.dat"
pairsfile=${data_file%.*}"-pairlist.txt"
 
#styles of NEB avail
single_zap="single_zap"
multi_zap="multi_zap"
multi_jump="multi_jump"
single_jump="single_jump"
boomerang="boomerang"


########################################################
########################################################
style=$single_zap
########################################################
########################################################

cyclelen=1
#lastdone="385 1189"
#alreadydone=1 #set to 1 to run everything



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

        num_repeat=5

        run_id="$atom_id"
        echo_string="to jump "$run_id" to BC of "$atomF1"-"$atomF2
        create_gif=true 
        

    fi

    
    for etol in 7e-6 #7e-6  #3e-7 1e-7 # 1e-5 #3e-6 1e-6 7e-7 5e-7 3e-7 1e-7  #7e-5 5e-5 3e-5 1e-5 
    do 
        for timestep in 0.5 #$(seq 1.6 0.1 1.8) 
        do
            
            ((numruns++))



            unique_tag=$run_id"_"$(date +%H%M%S)

            cwd=$(pwd) #current working directory
            out_folder=$cwd"/output/neb"${unique_tag}"/"
            data_folder=$out_folder"/logs/"

            mkdir -p $cwd"/output/" #just in case output folder is not made
            mkdir $out_folder #Now make folder where all the output will go
            mkdir $data_folder


            
            
           
            # for((iboom=0;iboom<$num_repeat;++iboom))
            # do
            neb_info_file=$out_folder"nebinfo_"$iboom".txt"
            echo "----------------Prepping NEB "$echo_string" ----------------"
            mpirun -np 3 python3 /home/agoga/documents/code/topcon-md/py/PrepNEB.py \
            --out=$out_folder --etol=$etol --ts=$timestep --dfile=$data_file --plot=$plot --atomid=$atom_id --info=$neb_info_file \
            --style=$style --fposx=$fPosx --fposy=$fPosy --fposz=$fPosz --bc1=$atomF1 --bc2=$atomF2 --bclist="$jumpPairs" --repeat=$num_repeat --zapid=$zap_id #  
            
            
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
                    #dataf=$neb_identifier"_final"

                    log_file=$data_folder$logf
                    # final_data_file=$data_folder$dataf

                    echo "----------------Running NEB for "$neb_identifier" ----------------"
                    mpirun -np $num_replica --oversubscribe lmp_mpi -partition "$num_replica"x1 \
                        -nocite -log $log_file -in $neb_file -var maxneb $maxneb -var maxclimb $maxclimb -screen none \
                        -var output_folder $out_folder -var ts $timestep -var etol $etol  -var springconst $springconst -pscreen none \
                        -var identifier $neb_identifier -var h_id $h_id -var atom_id $neb_atom_id -var nebI $nebI -var nebF $nebF #-var dataf $final_data_file     
                        #$out_folder/screen \                
                fi

                # if [[ $style == $boomerang && $line == "initial_coords_"$neb_identifier* ]]
                # then
                #     nebline=($line)
                #     fPosx=${nebline[1]}
                #     fPosy=${nebline[2]}
                #     fPosz=${nebline[3]}
                #     echo "$fPosx $fPosy $fPosz"
                # fi

                
            done 3< "$neb_info_file"
            
            #     if [[ $style == $boomerang ]]
            #     then
            #         ifin=$(($num_replica-1))
            #         data_file="$final_data_file"."$ifin".dat
            #         echo $data_file
            #     fi
            # done

            echo "----------------Post NEB "$echo_string"  ----------------"
            python3 /home/agoga/documents/code/topcon-md/py/Process-NEB.py \
            --out=$out_folder --etol=$etol --ts=$timestep --nebfolder=$nebfolder --dfile=$data_file \
            --k=$springconst --plot=$plot --info=$neb_info_file --style=$style --gif=$create_gif --neblog=$log_file \
            --atomid=$atom_id --cylen=$cyclelen #--remove=$zap_id 
        
        done
    done
done

end=`date +%s`

runtime=$( echo "$end-$start" | bc -l)
runtimeMin=$( echo "$runtime/60" | bc -l)
runtimeAvg=$( echo "$runtimeMin/$numruns" | bc -l)
echo "Total runtime:" $runtimeMin"m"
echo "AVG run time:" $runtimeAvg"m"
