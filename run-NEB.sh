#!/bin/bash -l
#! -cwd
#! -j y
#! -S /bin/bash


lmppre="lmp/"
FILE="lmp/NEB.lmp"
filename=${FILE#"$lmppre"}


export OMP_NUM_THREADS=1

etol=0.01
timestep=1


skippes=1
numruns=0
start=`date +%s`
maxneb=3000
maxclimb=1000
springconst=1
plot=1
data_folder=/home/agoga/documents/code/topcon-md/data/neb #/pinhole-dump-files/"

setName=/FullSet
distFolder=$data_folder$setName"/"
nebfolder="/home/agoga/documents/code/topcon-md/neb-out/"
mkdir -p $nebfolder$setName


# for datafile in "$distFolder"/*.dat
# do
datafile="/home/agoga/documents/code/topcon-md/data/neb/pinholeCenter/Hcon-1500-695.dat"
pairsfile=${datafile%.*}"-pairlist.txt"
#     # datafile=$data_folder"SiOxNEB-NOH.data"
#     # # datafile="SiOxNEB-NOH.data"
#     PAIRSFILE=${datafile%.*}"-pairlist.txt"
#     echo $PAIRSFILE
#     
#     

#     lastdone="385 1189"
#     alreadydone=1 #set to 1 to run everything



#     mapfile -t pairs < $PAIRSFILE

mapfile -t pairs < $pairsfile
# atom_id=1642 #2041 #3898 #3339 #H #4090 #1649 #2776 #1659 
# # atomremove=1638
for pair in "${pairs[@]}" # #"5976 5979" #
do
    # if [[ $alreadydone  == 0  && $pair != $lastdone ]] ; then
    #     continue
    # else
    #     alreadydone=1
    # fi

    pariarray=($pair)


    
    neb_info_file="nebinfo.txt"


    atom_id=142 #${pariarray[0]} #${pair% *}
    zap_id=0 #$atom_id #${pariarray[1]}
    xpos=${pariarray[1]}
    ypos=${pariarray[2]}
    zpos=${pariarray[3]}

    #@TODO for OH complex migration
    # atom_id=${pariarray[0]} #${pair% *}
    # zap_id=${pariarray[1]}
    # hydrogen_id=${pariarray[2]}

    #atomremove=${pair#* }


    for etol in 7e-6 #5e-6 #3e-7 1e-7 # 1e-5 #3e-6 1e-6 7e-7 5e-7 3e-7 1e-7  #7e-5 5e-5 3e-5 1e-5 
    do 
        for timestep in 0.5 #$(seq 1.6 0.1 1.8) 
        do

        # for atomremove in 3090 #4929 #3715 # 3341 # 3880  #1548 1545 3955 3632 3599
        # do
            
            ((numruns++))
            name=${filename%.*}


            unique_tag=$atom_id"-"$zap_id"_"$timestep"-"$etol"_"$(date +%H%M%S)

            CWD=$(pwd) #current working directory
            out_folder=$CWD"/output/"${name}${unique_tag}"/"
            data_folder=$out_folder"/logs/"

            mkdir -p $CWD"/output/" #just in case output folder is not made
            mkdir $out_folder #Now make folder where all the output will go
            mkdir $data_folder


            in_file=$CWD"/"$FILE

            
            cp $in_file $out_folder


            s=$out_folder$NAME"_SLURM.txt"

            echo_string="to move "$atom_id" to "$zap_id

            #echo "----------------Prepping NEB to move "$atom_id" to "$xpos" "$ypos" "$zpos" ----------------"
            echo "----------------Prepping NEB "$echo_string"----------------"
            python3 /home/agoga/documents/code/topcon-md/py/PrepNEB.py $out_folder $etol $timestep $datafile $plot $atom_id $xpos $ypos $zpos #$hydrogen_id
            
            
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
                    log_file=$data_folder$logf

                    echo "----------------Running NEB "$echo_string"----------------"
                    mpirun -np 13 --oversubscribe lmp_mpi -partition 13x1 -nocite -log $log_file -in $out_folder$filename -var maxneb $maxneb -var maxclimb $maxclimb -var atom_id $neb_atom_id -var output_folder $out_folder -var nebI $nebI -var nebF $nebF -var identifier $neb_identifier -var etol $etol -var ts $timestep -var springconst $springconst -pscreen none -var h_id $h_id 
                fi
                
            done 3< "$out_folder$neb_info_file"

            echo "----------------Post NEB "$echo_string"----------------"
            python3 /home/agoga/documents/code/topcon-md/py/Process-NEB.py $out_folder $atom_id $etol $timestep $zap_id $nebfolder $datafile $springconst $plot $out_folder$neb_info_file
            

        done
    done
done

end=`date +%s`

runtime=$( echo "$end-$start" | bc -l)
runtimeMin=$( echo "$runtime/60" | bc -l)
runtimeAvg=$( echo "$runtimeMin/$numruns" | bc -l)
echo "Total runtime:" $runtimeMin"m"
echo "AVG run time:" $runtimeAvg"m"
