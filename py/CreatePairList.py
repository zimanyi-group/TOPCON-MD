import ase.io
import ase.neighborlist
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

from skspatial.objects import Line, Sphere

debugatom=4

def dprint(id,str):
    if id == debugatom:
        print(str)

def dist(pos1,pos2):
    return ((pos2[0]-pos1[0])**2+(pos2[1]-pos1[1])**2+(pos2[2]-pos1[2])**2)**0.5

def check_for_h(atoms,id,hlist):
    hDist=1.1#ang
    cur=atoms[id]
    
    for n in hlist:
        nei=atoms[n]
        neitype=nei.symbol
        if neitype == 'H':
            d=dist(cur.position,nei.position)
            if d <hDist:
                return 1  
    return 0


dfile="/home/agoga/documents/code/topcon-md/data/pinhole-dump-files/Hcon-1500-440.data"

dfile="/home/agoga/documents/code/topcon-md/data/SiOxNEB-H.data"
dfile="/home/agoga/documents/code/topcon-md/data/SiOxNEB-NOH.data"
with open(dfile) as lammps_dump_fl_obj:
    atoms = ase.io.read(lammps_dump_fl_obj,format="lammps-data",style='charge',units='real',sort_by_id=True)#, format="lammps-data", index=0)
    
r=atoms.get_positions()
numAtoms=len(r)
print(numAtoms)
print('loaded')



#Set the atom types correct so they show up as Si and O
anums=np.zeros(numAtoms)
anums[atoms.get_atomic_numbers()==1]=14
anums[atoms.get_atomic_numbers()==2]=8
anums[atoms.get_atomic_numbers()==3]=1
atoms.set_atomic_numbers(anums)

        
cut=np.zeros(numAtoms)
cut[atoms.get_atomic_numbers()==8]=1.5
cut[atoms.get_atomic_numbers()==14]=1.05
cut[atoms.get_atomic_numbers()==1]=0.01

nl=ase.neighborlist.NeighborList(cut,self_interaction=False)
nl.update(atoms)

SiRad=10
zmin=20
zmax=28

pairs=[]
counts=np.zeros(numAtoms)


for i in range(numAtoms):
    cur=atoms[i]
    curtype=cur.symbol
    cpos=cur.position
    
    
    dprint(i,f'Debug - zpos:{cpos[2]}')
    
    #don't look too close to the interface
    if cpos[2]<zmin or cpos[2]>zmax:
        continue
    
    if curtype == 'O':
        indices, offsets = nl.get_neighbors(i)
        
        #print(str(i+1)+ ' ' + str(indices))
        on=[]
        sin=[]
        hn=[]
        #create the list of oxygen and Si neighbors to run through
        for n in indices:
            nei=atoms[n]
            neitype=nei.symbol
            if neitype == 'O':
                on.append(n)
            elif neitype =='Si':
                sin.append(n)
            elif neitype =='H':
                hn.append(n)
        
        
        #@TODO not a good way to do this function because we can't check the neighbor as well
        # as we didn't make a h list in the next loop
        if check_for_h(atoms,i,hn) == 1:
            print(f"atom {i+1} has H too close")
            continue
        
        dprint(i,f"debug - {on}")
        #run through the oxygen neighbors
        for n in on:
            nei=atoms[n]
            npos=nei.position
            
            dprint(i,f"debug testing {n}")
                
            if npos[2]<zmin or npos[2]>zmax:
                dprint(i,f"debug {n} - z")
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
                    dprint(i,f'debug {n} - inter')
                    continue
                
                #if this didn't throw an exception then check if the Si is a neighbor of the other atom
                sineighs, offsets = nl.get_neighbors(s)
                if n not in sineighs:
                    dprint(i,f"debug {n} - neighs")
                    continue
                
                
                #good pair if it got this far
                #add this pair to the pair list
                p1=(i+1,n+1)
                p2=(n+1,i+1)
                
                
                
                if p1 not in pairs and p2 not in pairs:
                    dprint(i,f"debug {n} - success")
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

pairfile=dfile[:-5]+"-pairlist.txt"
with open(pairfile,"w") as tf:
    for p in pairs:
        tf.write(f"{p[0]} {p[1]}\n")
    


