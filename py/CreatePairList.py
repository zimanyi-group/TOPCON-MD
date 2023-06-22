import ase.io
import ase.neighborlist
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

from skspatial.objects import Line, Sphere


dfile="/home/agoga/documents/code/topcon-md/data/SiOxNEB-NOH.dump"

with open(dfile) as lammps_dump_fl_obj:
    atoms = ase.io.read(lammps_dump_fl_obj, format="lammps-dump-text", index=0)
    
r=atoms.get_positions()
numAtoms=len(r)
print(numAtoms)
print('loaded')



#Set the atom types correct so they show up as Si and O
anums=np.zeros(numAtoms)
anums[atoms.get_atomic_numbers()==1]=14
anums[atoms.get_atomic_numbers()==2]=8
atoms.set_atomic_numbers(anums)

        
cut=np.zeros(numAtoms)
cut[atoms.get_atomic_numbers()==8]=1.5
cut[atoms.get_atomic_numbers()==14]=1.05

nl=ase.neighborlist.NeighborList(cut,self_interaction=False)#{('H', 'H'): .1, ('H', 'He'): .1, ('He', 'He'): 3})
nl.update(atoms)

SiRad=10
zmin=20
zmax=28

pairs=[]
counts=np.zeros(numAtoms)
debugatom=3487
for i in range(numAtoms):
    cur=atoms[i]
    curtype=cur.symbol
    cpos=cur.position
    
    #don't look too close to the interface
    if cpos[2]<zmin or cpos[2]>zmax:
        continue
    
    if curtype == 'O':
        indices, offsets = nl.get_neighbors(i)
        
        #print(str(i+1)+ ' ' + str(indices))
        on=[]
        sin=[]
        #create the list of oxygen and Si neighbors to run through
        for n in indices:
            nei=atoms[n]
            neitype=nei.symbol
            if neitype == 'O':
                on.append(n)
            elif neitype =='Si':
                sin.append(n)
        
        if i == debugatom:
            print(f"debug - {on}")
        #run through the oxygen neighbors
        for n in on:
            nei=atoms[n]
            npos=nei.position
            if i == debugatom:
                print(f"debug testing {n}")
                
            if npos[2]<zmin or npos[2]>zmax:
                if i == debugatom:
                    print(f"debug {n} - z")
                
                continue
            
            line=Line(cur.position,nei.position)
            
            #check if any of the neighboring Si make this oxygen a good pair
            for s in sin:
                si=atoms[s]
                sphere=Sphere(si.position,SiRad)
            
                #check if there is a silicon 'in between' them
                try:
                    pa, pb = sphere.intersect_line(line)
                except:
                    if i == debugatom:
                        print(f'debug {n} - inter')
                    continue
                
                #if this didn't throw an exception then check if the Si is a neighbor of the other atom
                sineighs, offsets = nl.get_neighbors(s)
                if n not in sineighs:
                    if i == debugatom:
                        print(f"debug {n} - neighs")
                    continue

                #add this pair to the pair list
                p1=(i+1,n+1)
                p2=(n+1,i+1)
                
                
                
                if p1 not in pairs and p2 not in pairs:
                    if i == debugatom:
                        print(f"debug {n} - success")
                    counts[i]+=1
                    counts[n]+=1
                    pairs.append(p1)
                    continue
                    #print(f"Thats a good one {str(p1)}")
        
#print(pairs)
print(len(pairs))


ids = [j+1 for j in range(numAtoms) if counts[j]>6]
print(f"Ids with greater than 6 pairs: {str(ids)}")
ids = [j+1 for j in range(numAtoms) if counts[j]==6]
print(f"Ids with 6 pairs: {str(ids)}")
ids = [j+1 for j in range(numAtoms) if counts[j]==5]
print(f"Number of ids with 5 pairs: {str(len(ids))}")

pairfile="/home/agoga/documents/code/topcon-md/data/noHpairs-v1.txt"
with open(pairfile,"w") as tf:
    for p in pairs:
        tf.write(f"{p[0]} {p[1]}\n")
    


