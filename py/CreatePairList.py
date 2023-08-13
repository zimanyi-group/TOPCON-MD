#!/usr/bin/env python
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from skspatial.objects import Line, Sphere
import sys

import os
import shutil
import pandas as pd
from pathlib import Path 
import NEBTools as nt
import string 


lowerletters=list(string.ascii_lowercase)

def dprint(id,str):
        if id == debugatom:
            print(str)

def perp_interface(cur, neigh):
    planeDir=[0,0,1]
    vecBuffer=20
    
    ang = nt.angle_between((cur-neigh),planeDir)
    if abs(ang) > vecBuffer:
        return False
    else:
        return True

def parallel_interface(cur, neigh):
    planeDir=[0,0,1]
    vecBuffer=2
    #checking angle between pair sep vector and interface plane vector
    #so to be parallel the angle between needs to be roughly 90
    ang = 90 - nt.angle_between((cur-neigh),planeDir)
    if abs(ang) > vecBuffer:
        return False
    else:
        return True

def closer_pair(apos,bpos):
    i1=16
    i2=32
    az=apos[2]
    bz=bpos[2]
    if az < 8:
        az+=32
    if bz < 8:
        bz+=32
    alow=abs(az-i1)
    blow=abs(bz-i1)
    
    ahigh=abs(i2-az)
    bhigh=abs(i2-bz)
    
    amin = min(alow,ahigh)
    bmin = min(blow,bhigh)
    if amin < bmin:#a is closer to the lower interface
        return True
    else:
        return False
        

def create_pair_list(datapath, dfile, distDir,writefile=False,split=1):
    
    SiRad=10
    
    zlowrange=[18,20]
    zhighrange=[28,30]
    
    SiOBondOrder=nt.SiOBondOrder

    zmin=19
    zmax=28
    filename=dfile.removesuffix('.dat').removesuffix('.data').removesuffix('.dump')

    
    # with open(datapath+dfile) as lammps_dump_fl_obj:
    #     apos = ase.io.read(lammps_dump_fl_obj,format="lammps-data",style='charge',units='real',sort_by_id=True)#, format="lammps-data", index=0)
    
    #create_bond_file(datapath,dfile,bondfile)
    #atoms=read_bonds(datapath+'/scratchfolder/'+bondfile)
    (atoms, simbox) = nt.read_file_data_bonds(datapath,dfile)
    numAtoms=len(atoms.index)


    print(f'--------Running file: {dfile} with {numAtoms} atoms--------')

    pairs=[]
    counts=np.zeros(numAtoms+1)
    hClose=[]
    badO=[]

    Oatoms=atoms[atoms['type']=='O']

                
    #first run through and make a list of all O atoms that are bonded to a H or at too close to the interface
    for i, row in Oatoms.iterrows():
        curpos=row['pos']
        nindices = row['bonds']#[0] take only the indexes of the bonds not bond order
        hn=[]
        on=[]
        sin=[]
        zpos=curpos[2]
        
        if len(nindices)==0:
            print(f"O atom {i} has no neighbors!")
        #find all the H neighbors
        for ni in nindices:
            n=ni[0]
            bo=ni[1]
            neitype=atoms.at[n,'type']
            
            
            
            if neitype =='H':
                hn.append(n)
            elif neitype =='O':
                
                on.append(n)
            elif neitype =='Si':
                sin.append(n)
                
                
        # if len(sin)>0 and len(on)>0:
        #     print(f"O atom - {i} is bonded to a Si AND H.")        

        # if len(sin)==0:
        #     print(f"Oxygen atom {i} has no Si neighbors!")
            
        #if there are any H neighbors
        if len(hn) > 0:
            #print(f"atom {i+1} has H too close")
            hClose.append(i)
            badO.append(i)
            continue
        
        # if (zpos<zlowrange[0] or zpos>zhighrange[1]): #not ((zpos>zlowrange[0] and zpos<zlowrange[1]) or (zpos>zhighrange[0] and zpos<zhighrange[1]))
        #     badO.append(i)
        #     continue
        
        
    #run through
    for i, row in Oatoms.iterrows():
        cpos=row['pos']
        nbonds = row['bonds']
        dprint(i,f'Debug - zpos:{cpos[2]}')
        
        #if this oxygen is a bad then skip
        if i in badO:
            continue

        on=[]
        sin=[]

        
        #create the list of Si neighbors to run through
        for n in nbonds:
            ni=n[0]
            bo=n[1]
            
            if bo < SiOBondOrder:
                continue
            
            neitype=atoms.at[ni,'type']
            if neitype =='Si':
                sin.append(ni)
        
        dprint(i,f"debug - {on}")
        
        #run through the silicon bonds and fill a list with any oxygen 
        for si in sin:
            neibonds = atoms.at[si,'bonds']
            
            for neib in neibonds:
                nei=neib[0]
                neibo=neib[1]
                
                if neibo < SiOBondOrder:
                    continue
            
                #skip if the oxygen is bad or if this is the current oxygen already looking at
                if nei in badO or nei == i:
                    continue
                
                neitype = atoms.at[nei,'type']
                if neitype != 'O':
                    continue
        
        
                dprint(i,f"debug testing {n}")
                    
                    
                neipos= atoms.at[nei,'pos']
                
                
                #check if the pair vector is along the current direction we're picking pairs along
                # if not perp_interface(cpos,neipos):
                # if not parallel_interface(cpos,neipos):
                #     continue
                
                p1=(i,nei)
                p2=(nei,i)
                
                #good pair if it got this far
                #add this pair to the pair list              
                if p1 not in pairs and p2 not in pairs:
                    dprint(i,f"debug {nei} - success")
                    counts[i]+=1
                    counts[nei]+=1
                    if closer_pair(cpos,neipos):
                        pairs.append(p1)
                    else:
                        pairs.append(p2)
                    #print(f"Thats a good one {str(p1)}")


    # print(f"{len(hClose)} Oxygen have H that are too close")
    # ids = [j+1 for j in range(numAtoms) if counts[j]>6]
    # if len(ids)>0:
    #     print(f"Ids with greater than 6 pairs: {str(ids)}")
    # ids = [j+1 for j in range(numAtoms) if counts[j]==6]
    # if len(ids)>0:
    #     print(f"Ids with 6 pairs: {str(ids)}")
    # ids = [j+1 for j in range(numAtoms) if counts[j]==5]
    # if len(ids)>0:
    #     print(f"Number of ids with 5 pairs: {str(len(ids))}")

    npairs=np.array(pairs)
    print("Gotere")

    if writefile:
        
        if not os.path.exists(datapath+distDir):
            os.mkdir(datapath+distDir)
        
        numpairs=len(pairs)

        pairlists=np.array_split(npairs,split)

        for i in range(split):
            curlen=int(pairlists[i].size/2)#double counts cause pairs
            
            
            presplit=""
            if split>1:
                presplit=lowerletters[i]
                
            pairname=presplit+filename+"-pairlist.txt"
            
            pairfile=datapath+distDir+pairname

            shutil.copyfile(datapath+dfile, datapath+distDir+presplit+dfile)
            
            with open(pairfile,"w") as tf:
                for p in pairlists[i]:
                    tf.write(f"{p[0]} {p[1]}\n")

            print(f"{curlen} total pairs added to the file {pairname}.")
    
    return pairs

            
if __name__=='__main__':

    #current run defines
    debugatom=-1
    
    datapath="/home/agoga/documents/code/topcon-md/data/neb/"

    
    distDirectory='FullSet/'
    fd=datapath+distDirectory
    if not os.path.exists(fd):
        os.makedirs(fd)
        
    # f="1.6-381.dat"
    # create_pair_list(datapath,f,distDirectory,True)
    
    #dlist = ["Hcon-1500-0.data","Hcon-1500-110.data","Hcon-1500-220.data","Hcon-1500-330.data","Hcon-1500-440.data","Hcon-1500-550.data","Hcon-1500-695.data","Hcon-1500-880.data","Hcon-1500-990.data"]
    
    i=0
    for d in Path(datapath).glob('*.dat'):
        # if str(d).endswith("1.6-1.dat"):
        #     i+=1
        print(f"{str(i)}) {str(d)}")
        dfile=str(d).split('/')[-1]
        create_pair_list(datapath,dfile,distDirectory,False)
     

    
    # plt.rcParams["figure.autolayout"] = True
    
    #Need two lammps instances so that when removing an atom and minimizing we don't increase time for final NEB image minimization
    # L= lammps('mpi',["-log",f'{datapath}/CreateBonds.log'])
    # file=dlist[0]
    # bondfile=file[:-5]+".bonds"
    # create_bond_file(L,datapath,file,bondfile)
    # df=read_bonds(datapath+bondfile)
    # print(df)

