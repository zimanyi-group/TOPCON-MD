#!/bin/bash -l
#! -cwd
#! -j y
#! -S /bin/bash

# Author Adam Goga
################################################################################################################################################
# This is the main pipeline script which has a lot of different settings, most of which are straight forward. It is critical to have the right
# folder structure and output files in each segment of the pipeline. Loops through all the data file/pairlist pairs in the
# '$data_folder/$setName' folder and runs the pipeline on each.
# 
# PrepNEB.py creates a info file placed in the output folder which informs the rest of the pipeline about what NEB calculation to actually runs.
#
#
# The call to LAMMPS which actually runs the NEB calculation using lmp/NEB.lmp must have its log files output to the temporary output folder
# which will the be read in by the final python script which creates the datafile/gif after verification that the NEB calculation was successful
#
################################################################################################################################################
neb_file=/home/adam/code/topcon-md/lmp/NEB.lmp


export OMP_NUM_THREADS=1

#placeholder values that get overwritten
etol=0.01
timestep=1


num_replica=7

skippes=1
numruns=0 #couning the number of runs
start=`date +%s`

maxneb=3000
maxclimb=1000
springconst=452 #roughly 20eV/ang^2

#final plot/gif
plot=true
create_gif=true

data_folder=/home/adam/code/topcon-md/data/neb #/pinhole-dump-files/"
# data_folder=/home/adam/code/topcon-md/data/create_dat #/pinhole-dump-files/"


setName=/pair_lists/bc_to_bc/
distFolder=$data_folder$setName"/"

nebfolder="/home/adam/code/topcon-md/neb-out/hydrogen_project/"

mkdir -p $nebfolder$setName


#loop through all the datafile/pairlist sets
for data_file in "$distFolder"/*.dat
do
    #data_file="${data_folder}${setName}/1.6-143.dat"

    pairsfile=${data_file%.*}"-pairlist.txt"
    
################################
    # Styles of NEB avail
    # zap takes two atoms a mover and a zapped atom, the zapped atom is deleted and it's location is used as the mover atom's final location.
    # This can be done with multiple atoms, for example moving a OH complex to a nearby O vacancy
    single_zap="single_zap"
    multi_zap="multi_zap"

    # This is a test NEB process, it takes a H atom and a final location. PrepNEB will then create another H atom in the region near that final
    # location. The final NEB image is the initial H atom and this new atom in a formed H2 bond.
    h_to_h2="h_to_h2"

    # Multi/single jump take a atom and a location and moves the atom to the location or multiple locations. Multi-jump will perform a number 
    # of NEB calculations equal to the number of final locations given, and each final location will be the initial location for the next jump.
    multi_jump="multi_jump"
    single_jump="single_jump"

    # Interstitial was a test function which searched an area nearby a given atom for a viable interstitial location to move to. This is better
    # done in the CreatePairList.py step of the pipeline but it is left for posterity.
    interstitial="interstitial"

    # Boomerang is a style of the pipeline which moves the atom from initial to final location multiple times over and over again. This produces 
    # interesting results and required some testing to determine if it was more realistic than the normal NEB, but was ultimately not used.
    boomerang="boomerang"
    boomerang_zap="boomerang_zap"
################################

    ########################################################
    ########################################################
    style=$single_jump
    ########################################################
    ########################################################

    cyclelen=1
    num_boomerang=1

    #lastdone="385 1189"
    #alreadydone=1 #set to 1 to run everything



    mapfile -t pairs < $pairsfile

    echo $pairs
    for pair in "${pairs[@]}" #"5976 5979" #
    do

        pairarray=($pair)

        echo $pairarray
        #Single zap
        if [[ $style == $single_zap ]];then

            atom_id=${pairarray[0]} 
            zap_id=${pairarray[1]}

            run_id="$atom_id-$zap_id"
            echo_string="to zap from "$run_id
            create_gif=true
            

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
            
            # atomF1=${pairarray[4]}
            # atomF2=${pairarray[5]}

            run_id="$atom_id"
            echo_string="to jump "$run_id" to loc - "$fPosx","$fPosy","$fPosz
            create_gif=true 
        elif [[ $style == $h_to_h2 ]];then

            atom_id=${pairarray[0]} 
            fPosx=${pairarray[1]}
            fPosy=${pairarray[2]}
            fPosz=${pairarray[3]}
            # h2x=${pairarray[4]}
            # h2y=${pairarray[5]}
            # h2z=${pairarray[6]}
            

            run_id="$atom_id"
            echo_string="to jump "$run_id" to loc - "$fPosx","$fPosy","$fPosz #creating H at "$h2x","$h2y","$h2z
            create_gif=true 

        elif [[ $style == $boomerang ]];then
            cyclelen=2
            atom_id=${pairarray[0]} 
            fPosx=${pairarray[1]}
            fPosy=${pairarray[2]}
            fPosz=${pairarray[3]}
            

            num_repeat=$num_boomerang

            run_id="$atom_id"
            echo_string="to boomerang "$run_id" to loc - "$fPosx","$fPosy","$fPosz" "$num_repeat" times"
            create_gif=true 

        elif [[ $style == $boomerang_zap ]];then
            cyclelen=2
            atom_id=${pairarray[0]} 

            zap_id=${pairarray[1]}

            run_id="$atom_id-$zap_id"
            echo_string="to zap from "$run_id

            num_repeat=1

            create_gif=true 
            
        elif [[ $style == $interstitial ]];then

            atom_id="${pairarray[0]}${pairarray[1]}"
            zap_id=

            run_id="$atom_id"
            echo_string="interstitial w/ seed "$run_id


            create_gif=true 
        fi

        
        for etol in 7e-6 #7e-6  #3e-7 1e-7 # 1e-5 #3e-6 1e-6 7e-7 5e-7 3e-7 1e-7 
        do 
            for timestep in 0.5 #$(seq 1.6 0.1 1.8) 
            do
                
                ((numruns++))


                good_run=false
                unique_tag=$run_id"_"$(date +%H%M%S)

                cwd=$(pwd) #current working directory
                out_folder=$cwd"/output/neb"${unique_tag}"/"
                data_folder=$out_folder"/logs/"

                mkdir -p $cwd"/output/" #just in case output folder is not made
                mkdir $out_folder #Now make folder where all the output will go
                mkdir $data_folder

            

                neb_info_file=$out_folder"nebinfo_"$run_id".txt"

                echo "----------------Prepping NEB "$echo_string" ----------------"
                mpiexec -np 1 python3 /home/adam/code/topcon-md/py/PrepNEB.py \
                --out=$out_folder --etol=$etol --ts=$timestep --dfile=$data_file --plot=$plot --atomid=$atom_id --info=$neb_info_file \
                --style=$style --fposx=$fPosx --fposy=$fPosy --fposz=$fPosz --h2x=$h2x --h2y=$h2y --h2z=$h2z --bc1=$atomF1 --bc2=$atomF2 --bclist="$jumpPairs" --repeat=$num_repeat --zapid=$zap_id #  
                

                #now ready the info file that was just created by PrepNEB.py and for every line that starts with neb, run a neb calculation with the input parameters
                while read -u3 line
                do
                    #echo $line
                    
                    if [[ $line == neb* ]] #if the line starts with "neb"
                    then
                        good_run=true
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
                        mpiexec -np 14 lmp_mpi -partition "$num_replica"x2 \
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

                if [[ $good_run == true ]]
                then
                    echo "----------------Post NEB "$echo_string"  ----------------"
                    python3 /home/adam/code/topcon-md/py/Process-NEB.py \
                    --out=$out_folder --etol=$etol --ts=$timestep --nebfolder=$nebfolder --dfile=$data_file \
                    --k=$springconst --plot=$plot --info=$neb_info_file --style=$style --gif=$create_gif --neblog=$log_file \
                    --atomid=$neb_atom_id --cylen=$cyclelen #--remove=$zap_id 
                fi
            
            done
        done
    done
done

end=`date +%s`

runtime=$( echo "$end-$start" | bc -l)
runtimeMin=$( echo "$runtime/60" | bc -l)
runtimeAvg=$( echo "$runtimeMin/$numruns" | bc -l)
echo "Total runtime:" $runtimeMin"m"
echo "AVG run time:" $runtimeAvg"m"
