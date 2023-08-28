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

from mpi4py import MPI
me = MPI.COMM_WORLD.Get_rank()
numproc=MPI.COMM_WORLD.Get_size()


lowerletters=list(string.ascii_lowercase)

lowerletters=list(string.ascii_lowercase)

def dprint(id,str):
        if id == debugatom:
            print(str)

def perp_interface(cur, neigh):
    planeDir=[0,0,1]
    vecBuffer=20
    vecBuffer=20
    
    ang = nt.angle_between((cur-neigh),planeDir)
    ang = nt.angle_between((cur-neigh),planeDir)
    if abs(ang) > vecBuffer:
        return False
    else:
        return True

def parallel_interface(cur, neigh):
    planeDir=[0,0,1]
    vecBuffer=2
    vecBuffer=2
    #checking angle between pair sep vector and interface plane vector
    #so to be parallel the angle between needs to be roughly 90
    ang = 90 - nt.angle_between((cur-neigh),planeDir)
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
    zlowrange=[18,20]
    zhighrange=[28,30]
    
    SiOBondOrder=nt.SiOBondOrder
    SiOBondOrder=nt.SiOBondOrder

    zmin=19
    zmin=19
    zmax=28
    filename=dfile.removesuffix('.dat').removesuffix('.data').removesuffix('.dump')

    
    # with open(datapath+dfile) as lammps_dump_fl_obj:
    #     apos = ase.io.read(lammps_dump_fl_obj,format="lammps-data",style='charge',units='real',sort_by_id=True)#, format="lammps-data", index=0)
    filename=dfile.removesuffix('.dat').removesuffix('.data').removesuffix('.dump')

    
    # with open(datapath+dfile) as lammps_dump_fl_obj:
    #     apos = ase.io.read(lammps_dump_fl_obj,format="lammps-data",style='charge',units='real',sort_by_id=True)#, format="lammps-data", index=0)
    
    #create_bond_file(datapath,dfile,bondfile)
    #atoms=read_bonds(datapath+'/scratchfolder/'+bondfile)
    (atoms, simbox) = nt.read_file_data_bonds(datapath,dfile)
    numAtoms=len(atoms.index)

    #create_bond_file(datapath,dfile,bondfile)
    #atoms=read_bonds(datapath+'/scratchfolder/'+bondfile)
    (atoms, simbox) = nt.read_file_data_bonds(datapath,dfile)
    numAtoms=len(atoms.index)


    print(f'--------Running file: {dfile} with {numAtoms} atoms--------')

    pairs=[]
    counts=np.zeros(numAtoms+1)
    counts=np.zeros(numAtoms+1)
    hClose=[]
    badO=[]

    Oatoms=atoms[atoms['type']=='O']

    numoh=0
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
                
                
        if len(sin)==1 and len(hn)==1:
            # print(f"O atom - {i} is bonded to a Si AND H.")  
            numoh+=1
            
              

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
    print(numoh)
        
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


    if writefile and me==0:
        
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

def create_oh_pair_list(datapath, dfile, distDir,writefile=False,split=1):
    
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

    numoh=0
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
                
                
        if len(sin)==1 and len(hn)==1:
            print(f"O atom - {i} is bonded to a Si AND H.")  
            numoh+=1
        else:
            badO.append(i)
            
    print(f"Number of viable OH: {numoh}")
        
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
        hi=None

        
        #create the list of Si and H neighbors to run through
        for n in nbonds:
            ni=n[0]
            bo=n[1]
            
    
            neitype=atoms.at[ni,'type']
            if neitype =='H':
                if hi == None:
                    hi=ni
                else:
                    print("WARNING FOUND TWO H WHEN THERE SHOULD ONLY BE ONE ")
            if neitype =='Si':
                if bo < SiOBondOrder:
                    continue
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
            for neib in neibonds:
                nei=neib[0]
                neibo=neib[1]
                
                if neibo < SiOBondOrder:
                    continue
            
                #skip if the oxygen is bad or if this is the current oxygen already looking at
                # if nei in badO or nei == i:
                #     continue
                
                if nei ==i:
                    continue
                
                neitype = atoms.at[nei,'type']
                if neitype != 'O':
                    continue
        
        
                dprint(i,f"debug testing {n}")
                    
                    
                neipos= atoms.at[nei,'pos']
                    
                neipos= atoms.at[nei,'pos']
                
  
                p1=(i, nei, hi)

                
                #good pair if it got this far
                #add this pair to the pair list              
                if p1 not in pairs:
                    dprint(i,f"debug {nei} - success")
                    counts[i]+=1
                    counts[nei]+=1
                    
                    pairs.append(p1)


    npairs=np.array(pairs)

    if writefile and me==0:
        
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
                    tf.write(f"{p[0]} {p[1]} {p[2]}\n")

            print(f"{curlen} total pairs added to the file {pairname}.")
    
    return pairs

def create_pinhole_pair_list_edge(datapath, dfile, distDir, pinholeCenter, writefile=False,split=1):
    
    #togeather with the pinhole center, these define a shell which we can pick Oxygen's from
    minRad=7
    maxRad=12
    
    #how far to move the O atoms
    movedist=3
    SiOBondOrder=nt.SiOBondOrder

    ifacez=15
    maxz=27
    zmin=5

    filename=dfile.removesuffix('.dat').removesuffix('.data').removesuffix('.dump')

    (atoms, simbox) = nt.read_file_data_bonds(datapath,dfile)
    numAtoms=len(atoms.index)
    

    print(f'--------Running file: {datapath+dfile} with {numAtoms} atoms--------')

    pairs=[]



    Oatoms=atoms[atoms['type']=='O']

    numoh=0
    #first run through and make a list of all O atoms that are bonded to a H or at too close to the interface
    for i, row in Oatoms.iterrows():
        curpos=row['pos']
        nindices = row['bonds']#[0] take only the indexes of the bonds not bond order
        
        zp=curpos[2]
    
        if zp < ifacez+movedist or zp > maxz :
            continue
        
        pinholeCenterCurZ=pinholeCenter
        pinholeCenterCurZ[2]=zp
        sepVec=nt.pbc_vec_subtract(simbox,pinholeCenterCurZ,curpos)
        
        
    
        dist=nt.pbc_dist(simbox,pinholeCenterCurZ,curpos)

            
        sepVecN=sepVec/np.linalg.norm(sepVec)

        if dist > minRad and dist < maxRad:
            
        
            hn=[]
            on=[]
            sin=[]
            
            #find all the H neighbors
            for ni in nindices:
                n=ni[0]
                
                neitype=atoms.at[n,'type']
                
                
                if neitype =='H':
                    hn.append(n)
                elif neitype =='O':
                    on.append(n)
                elif neitype =='Si':
                    sin.append(n)
            
            if len(hn) > 0:
                continue
            
            fpos=curpos+ sepVecN*movedist
            
            neighvac=find_neighboring_sibc(atoms,i)
            
            
            
            if len(neighvac) ==0:
                continue
            
            mindist=100
            minpos=None
            minvac=None
            for vac in neighvac:
                p1=atoms.at[vac[0],'pos']
                p2=atoms.at[vac[1],'pos']
                
                #get the midpoint of the neighboring vacancy
                midpt=p1+np.array(nt.pbc_vec_subtract(simbox,p1,p2))/2
            
                #get the seperation vector between the vacancy and the attempted positon
                sep=nt.pbc_vec_subtract(simbox,midpt,fpos)
                
                vacdist=np.linalg.norm(sep)
                #print(f"       poslist for {vac}: {curpos} {p1} {p2}\n       mid:{midpt} dist from guess:{vacdist}")
                if vacdist < mindist:
                    minvac=vac
                    minpos=midpt
                    mindist=vacdist
            
            
            
            #pos=nt.find_local_minima_position(datapath+dfile,i,minvac)
            
            #now check if the vacancy is on the 'get out of pinhole' path
            
            
            
            ang=nt.angle_between_pts(fpos,minpos,curpos,simbox)[0][0]
            if ang > 50:
                #print(f"-----FAILED Atom {i} with ang={ang}, pos: {curpos}------\n------Best vacancy: {minvac} with dist={mindist}-----")
                continue
            
            pl=[i,minpos]
            print(f"-----Atom {i} with ang={ang}, pos: {curpos} dist to pinhole:{dist}------\n------Best vacancy: {minvac} with dist={mindist}, adding {pl}------")
            pairs.append(pl)
            

    npairs=np.array(pairs,dtype=object)

    if writefile and me==0:
        
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
                    #tf.write(f"{p[0]} {p[1]}\n")#[0]} {p[1][1]} {p[1][2]}\n")
                    tf.write(f"{p[0]} {p[1][0]} {p[1][1]} {p[1][2]}\n")
            print(f"{curlen} total pairs added to the file {pairname}.")
    
    return pairs

def create_pinhole_center_out_pair_list(datapath, dfile, distDir, pinholeCenter, writefile=False,split=1):
    
    #togeather with the pinhole center, these define a shell which we can pick Oxygen's from
    minRad=7
    maxRad=12
    
    #how far to move the O atoms
    movedist=3
    SiOBondOrder=nt.SiOBondOrder

    ifacez=15
    zmin=5

    filename=dfile.removesuffix('.dat').removesuffix('.data').removesuffix('.dump')

    (atoms, simbox) = nt.read_file_data_bonds(datapath,dfile)
    numAtoms=len(atoms.index)
    

    print(f'--------Running file: {datapath+dfile} with {numAtoms} atoms--------')

    pairs=[]



    Oatoms=atoms[atoms['type']=='O']

    numoh=0
    #first run through and make a list of all O atoms that are bonded to a H or at too close to the interface
    for i, row in Oatoms.iterrows():
        curpos=row['pos']
        nindices = row['bonds']#[0] take only the indexes of the bonds not bond order
        
        zp=curpos[2]
    
        if zp < ifacez+movedist:
            continue
        
        sepVec=nt.pbc_vec_subtract(simbox,pinholeCenter,curpos)
        
        
        #distance from pinhole center
        dist=nt.pbc_dist(simbox,pinholeCenter,curpos)

        
        #@TODO  fucking ugly
        sepVecN=np.zeros(3)
        
        for j in range(len(sepVec)):
            sepVecN[j]=sepVec[j]/dist
            
            
            
        if dist > minRad and dist < maxRad:
            
        
            hn=[]
            on=[]
            sin=[]
            
            #find all the neighbors
            for ni in nindices:
                n=ni[0]
                
                neitype=atoms.at[n,'type']
                
                
                if neitype =='H':
                    hn.append(n)
                elif neitype =='O':
                    on.append(n)
                elif neitype =='Si':
                    sin.append(n)
            
            if len(hn) > 0:
                continue
            
            
            neisibc=find_neighboring_sibc(atoms,i)
            print(neisibc)
            fpos=curpos+ sepVecN*movedist
            print(i)
            #pos=nt.find_local_minima_position(datapath+dfile,i,fpos)
            #pl=[i,pos]
            
            
            # print(pl)
            # pairs.append(pl)


    npairs=np.array(pairs,dtype=object)

    if writefile and me==0:
        
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
                    #tf.write(f"{p[0]} {p[1]}\n")#[0]} {p[1][1]} {p[1][2]}\n")
                    tf.write(f"{p[0]} {p[1][0]} {p[1][1]} {p[1][2]}\n")
            print(f"{curlen} total pairs added to the file {pairname}.")
    
    return pairs

def find_nearby_O():
    distdf=nt.apply_dist_from_pos(atoms,simbox,fpos,"O")         
    # mdv2=movedist*.75
    # distdf=distdf[distdf["dist"]<mdv2]

    # if len(distdf)==0:
    #     continue
    
    # (pvdistdf,pvdcol)=nt.apply_point_vec_dist(distdf,simbox,curpos,fpos,'O')
    # pvdistdf=pvdistdf[pvdistdf[pvdcol]<mdv2]
    # lowest=i
    # #print(pvdistdf.to_string())
    
    
    # #Try the lowest distance from the point we want to be at and nearby Oxygens positon
    # while lowest == i:
    #     lowest=pvdistdf[pvdcol].idxmin()
    #     # if lowest==i:
    #     #     print(distdf.to_string())
    #     pvdistdf.drop(index=lowest,inplace=True)
    #     pl=[i,lowest]
    #     lpos=atoms.at[lowest,'pos']
        
    #     #SKIP if this current atom is closer than our desired position 
    #     pdcl=nt.pbc_dist(simbox,curpos,lpos)
    #     pdi=nt.pbc_dist(simbox,pinholeCenter,curpos)
    #     pdl=nt.pbc_dist(simbox,pinholeCenter,lpos) 
    #     if pdcl < movedist or pdcl > 2*movedist or pdl < pdi:
    #         #print(f"Tried:{fpos}, got:{lpos}")
    #         lowest=i
            
    #         if len(pvdistdf)==0:
    #             break
    
    
    
    # if lowest == i:
    #     print(f"Bad {i}")
    #     continue
    # lpos=atoms.at[lowest,'pos']
        
    
    # #if the final location is closer to the pinhole than the initial, don't add it
    # pdi=nt.pbc_dist(simbox,pinholeCenter,curpos)
    # pdl=nt.pbc_dist(simbox,pinholeCenter,lpos)
    # if pdl < pdi:
    #     print(f"Bad {i}- Tried:{fpos}")
    #     continue#if the new place is closer to the pinhole center then skip
    
    # md=nt.pbc_dist(simbox,curpos,lpos)
    # print(f"Good {i}-  {md} from {fpos} to {lpos} - zp {zp}")
    
    
def find_neighboring_sibc(atoms,oi):
    #start with an Oxygen and find all neighboring Si with BC vacancies
    nindices=atoms.at[oi,'bonds']
    #look to see if this atom has any neighboring SI with vacancies

    si_bc_vac=[]
    badsi=[]
    for n in nindices:
        
        ni=n[0]
        neitype=atoms.at[ni,'type']
        if neitype=='Si':

            sibonds=atoms.at[ni,'bonds']
                    
            badsi=[]  
            for b in sibonds:
                bi=b[0]
                btype=atoms.at[bi,'type']
                if btype=='H' or btype=='O':
                    bb=atoms.at[bi,'bonds']
                    for bn in bb:
                        bni=bn[0]
                        bntype=atoms.at[bni,'type']
                        if bntype=='Si' and bni !=ni:
                            badsi.append(bni)
        
            
            for nnn in sibonds:
                nnni=nnn[0]
                nnntype=atoms.at[nnni,'type']
                if nnntype=='Si'and nnni not in badsi:
                    si_bc_vac.append([ni,nnni])
    
    return si_bc_vac
          
def find_neighboring_sibc_recursion(atoms,si):          
    #start with a silicon and find all bc vacancies chains it has recursively
    si_bc_vac=[]
    sibonds=atoms.at[si,'bonds']
            
    badsi=[]  
    for b in sibonds:
        bi=b[0]
        btype=atoms.at[bi,'type']
        if btype=='H' or btype=='O':
            bb=atoms.at[bi,'bonds']
            for bn in bb:
                bni=bn[0]
                bntype=atoms.at[bni,'type']
                if bntype=='Si' and bni !=si:
                    badsi.append(bni)
                    
                
    
    for nnn in sibonds:
        nnni=nnn[0]
        nnntype=atoms.at[nnni,'type']
        if nnntype=='Si'and nnni not in badsi:
            si_bc_vac.append([si,nnni])
    
    return si_bc_vac

def find_neighboring_bc_chains(atoms,curatom):
    #start with an Oxygen and find all neighboring Si with BC vacancies
    nindices=atoms.at[curatom,'bonds']
    #look to see if this atom has any neighboring SI with vacancies
    si_bc_vac_chains=[]
    si_bc_vac=[]
    badsi=[]
    for n in nindices:
        
        ni=n[0]
        
        neitype=atoms.at[ni,'type']
        if neitype=='Si':
            
            si_bc_vac=find_neighboring_sibc(atoms,ni)
            
            
            for vac in si_bc_vac:
                nextsi=vac[1]
                
                while True:
                    cur_chain=[]
                    cur_chain.append(vac)
                    nextchain=find_neighboring_sibc(atoms,nextsi)
                    
                    
                    if len(nextchain) ==0:
                        break
            
    return si_bc_vac_chains
        
    

if __name__=='__main__':

    #current run defines
    debugatom=-1
    
    datapath="/home/agoga/documents/code/topcon-md/data/neb/"
    datapath="/home/agoga/documents/code/topcon-md/data/neb/"

    
    distDirectory='pinholeCenter/'
    fd=datapath+distDirectory
    if not os.path.exists(fd):
        os.makedirs(fd)
        
    # f="1.6-381.dat"
    # create_pair_list(datapath,f,distDirectory,True)
    # f="1.6-381.dat"
    # create_pair_list(datapath,f,distDirectory,True)
    
    #dlist = ["Hcon-1500-0.data","Hcon-1500-110.data","Hcon-1500-220.data","Hcon-1500-330.data","Hcon-1500-440.data","Hcon-1500-550.data","Hcon-1500-695.data","Hcon-1500-880.data","Hcon-1500-990.data"]
    #dlist = ["Hcon-1500-0.data","Hcon-1500-110.data","Hcon-1500-220.data","Hcon-1500-330.data","Hcon-1500-440.data","Hcon-1500-550.data","Hcon-1500-695.data","Hcon-1500-880.data","Hcon-1500-990.data"]
    
    i=1
    for d in Path(fd).glob('*.dat'):
        if not str(d).endswith("Hcon-1500-695.dat"):
            continue
        print(f"{str(i)}) {str(d)}")
        dfile=str(d).split('/')[-1]
        create_pinhole_pair_list_edge(datapath,dfile,distDirectory,[27,27,20],True)
        i+=1
     
     
     
     #Hcon-1500-695 - pinhole "center" ~ (21,10,22)

