#!/bin/bash
setName=FullSet/
baseFolder=/home/agoga/sandbox/topcon/
dataFolder=data/neb/$setName
distFolder=$baseFolder$dataFolder

for DATAFILE in "$distFolder"/*.dat
do
    echo "$dataFolder${DATAFILE##*/}"
    sbatch farm-run-neb.sh $dataFolder${DATAFILE##*/} $setName
done



# #!/bin/bash
# distFolder=/home/agoga/sandbox/topcon/data/SiOxVaryH

# for DATAFILE in "$distFolder"/*.data 
# do
#     echo "1"
#     #sbatch farm-run-createdat.sh data/SiOxVaryH/${DATAFILE##*/}
# done





