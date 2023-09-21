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
plot=true
create_gif=true
data_folder=/home/agoga/documents/code/topcon-md/data/neb #/pinhole-dump-files/"

setName=/pinholeCenter
distFolder=$data_folder$setName"/"
nebfolder="/home/agoga/documents/code/topcon-md/neb-out/"
mkdir -p $nebfolder$setName


# for datafile in "$distFolder"/*.dat
# do
datafile="/home/agoga/documents/code/topcon-md/data/neb/pinholeCenter/Hcon-1500-695.dat"
pairsfile=${datafile%.*}"-pairlist.txt"
 

#lastdone="385 1189"
#alreadydone=1 #set to 1 to run everything



# mapfile -t pairs < $pairsfile

# for pair in "${pairs[@]}" # #"5976 5979" #
# do
# if [[ $alreadydone  == 0  && $pair != $lastdone ]] ; then
#     continue
# else
#     alreadydone=1
# fi

pariarray=($pair)


datafile=$distFolder"Hcon-1500-695.dat"

#${pair% *}
# fPosx=38.856
# fPosy=32.47
# fPosz=13.9895
atomremove=0


atom_id=822 
jumpPairs="480 909,909 414,414 893,893 1321,1320 1321,1320 532"


atom_id=1319 
jumpPairs="308 1299,1299 437,437 309,309 461"






# atom_id=790
# j1="791 766"
# j2="766 1197"
# j3="1197 1196"
# j0="786 791"
# jumpPairs="$j1,$j2,$j3,$j2,$j1,$j0"



atom_id=4657
jumpPairsBase="1225 3163,3163 4654,4654 3856,3163 4654,1225 3163,3161 1225"


#ends at an already 4 coordinated Si
atom_id=3864
jumpPairsBase="577 1232,1232 3161,3161 3862,1232 3161,577 1232,577 4286"


atom_id=388 #non-converging
jumpPairsBase="1339 359,359 1342,1342 397,359 1342,1339 359,361 1339"
# forward only
# atom_id=1257 
# jumpPairsBase="172 826,172 820,820 4203,4203 6125,6125 4924,4924 5450,5450 5866,5866 5347"

# atom_id=790
# j1="791 766"
# j2="766 1197"
# j3="1197 1196"
# j0="786 791"
# jumpPairs="$j1,$j2,$j3,$j2,$j1,$j0"
jumpPairs=$jumpPairsBase
for ((i=1;i<1;i+=1))
do
    jumpPairs="$jumpPairs,$jumpPairsBase"
done

style="multi_jump"
#atomremove=${pair#* }


for etol in 7e-6 #7e-6  #3e-7 1e-7 # 1e-5 #3e-6 1e-6 7e-7 5e-7 3e-7 1e-7  #7e-5 5e-5 3e-5 1e-5 
do 
    for timestep in 0.5 #$(seq 1.6 0.1 1.8) 
    do
        
        ((numruns++))
        name=${filename%.*}


        unique_tag=$atom_id"-_"$timestep"-"$etol"_"$(date +%H%M%S)

        CWD=$(pwd) #current working directory
        out_folder=$CWD"/output/"${name}${unique_tag}"/"
        data_folder=$out_folder"/logs/"

        mkdir -p $CWD"/output/" #just in case output folder is not made
        mkdir $out_folder #Now make folder where all the output will go
        mkdir $data_folder


        in_file=$CWD"/"$FILE

        
        cp $in_file $out_folder


        s=$out_folder$NAME"_SLURM.txt"

        neb_info_file=$out_folder"nebinfo.txt"
        echo_string="to multijump "$atom_id

        echo "----------------Prepping NEB "$echo_string" ----------------"
        mpirun -np 4 python3 /home/agoga/documents/code/topcon-md/py/PrepNEB.py \
        --out=$out_folder --etol=$etol --ts=$timestep --dfile=$datafile --plot=$plot --atomid=$atom_id --info=$neb_info_file \
        --style=$style --bclist="$jumpPairs" 
        
        
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

                echo "----------------Running NEB for "$neb_identifier" ----------------"
                mpirun -np 13 --oversubscribe lmp_mpi -partition 13x1 \
                    -nocite -log $log_file -in $out_folder$filename -var maxneb $maxneb -var maxclimb $maxclimb \
                    -var output_folder $out_folder -var ts $timestep -var etol $etol  -var springconst $springconst -pscreen $out_folder/screen \
                    -var identifier $neb_identifier -var h_id $h_id -var atom_id $neb_atom_id -var nebI $nebI -var nebF $nebF                      
            fi

            
        done 3< "$neb_info_file"

        echo "----------------Post NEB "$echo_string"  ----------------"
        python3 /home/agoga/documents/code/topcon-md/py/Process-NEB.py \
        --out=$out_folder --atomid=$atom_id --etol=$etol --ts=$timestep --remove=$atomremove --nebfolder=$nebfolder --dfile=$datafile \
        --k=$springconst --plot=$plot --info=$neb_info_file --style=$style --gif=$create_gif
    
    done
done
# done

end=`date +%s`

runtime=$( echo "$end-$start" | bc -l)
runtimeMin=$( echo "$runtime/60" | bc -l)
runtimeAvg=$( echo "$runtimeMin/$numruns" | bc -l)
echo "Total runtime:" $runtimeMin"m"
echo "AVG run time:" $runtimeAvg"m"