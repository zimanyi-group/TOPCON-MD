#!/usr/bin/env python
from lammps import lammps
import ase.io
import ase.neighborlist
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from skspatial.objects import Line, Sphere
import sys
from math import degrees
import os
import shutil
import pandas as pd


def dprint(id,str):
        if id == debugatom:
            print(str)

def dist(pos1,pos2):
    return ((pos2[0]-pos1[0])**2+(pos2[1]-pos1[1])**2+(pos2[2]-pos1[2])**2)**0.5

def check_for_h(atoms,id,hlist):
    hDist=1.05#ang
    cur=atoms[id]
    
    for n in hlist:
        nei=atoms[n]
        neitype=nei.symbol
        if neitype == 'H':
            d=dist(cur.position,nei.position)
            if d < hDist:
                return True
    return False

def position_check(zpos, zlowrange, zhighrange):
    #return not ((zpos>zlowrange[0] and zpos<zlowrange[1]) or (zpos>zhighrange[0] and zpos<zhighrange[1]))
    return (zpos<zlowrange[0] or zpos>zhighrange[1])

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    if v1 is None or v2 is None:
        return 2*np.pi
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def perp_interface(cur, neigh):
    planeDir=[0,0,1]
    vecBuffer=40
    
    ang = angle_between((cur.position-neigh.position),planeDir)
    if abs(ang) > vecBuffer:
        return False
    else:
        return True

def parallel_interface(cur, neigh):
    planeDir=[0,0,1]
    vecBuffer=1
    #checking angle between pair sep vector and interface plane vector
    #so to be parallel the angle between needs to be roughly 90
    ang = 90 - angle_between((cur.position-neigh.position),planeDir)
    if abs(ang) > vecBuffer:
        return False
    else:
        return True

def create_pair_list(datapath, dfile, distDir):
    
    SiRad=10
    
    zlowrange=[17,20]
    zhighrange=[28,31]
    

    zmin=20
    zmax=28
    
    with open(datapath+dfile) as lammps_dump_fl_obj:
        atoms = ase.io.read(lammps_dump_fl_obj,format="lammps-data",style='charge',units='real',sort_by_id=True)#, format="lammps-data", index=0)
        
    r=atoms.get_positions()
    numAtoms=len(r)

    print(f'--------Running file: {dfile} with {numAtoms} atoms--------')



    #Set the atom types correct so they show up as Si and O
    anums=np.zeros(numAtoms)

    #here wea re setting atomic numbers of the different types
    anums[atoms.get_atomic_numbers()==1]=14
    anums[atoms.get_atomic_numbers()==2]=8
    anums[atoms.get_atomic_numbers()==3]=1
    atoms.set_atomic_numbers(anums)

    #here we set the cut off radius for each atom type in angstroms
    #If two atom's circles overlap then they are neighbors
    cut=np.zeros(numAtoms)
    cut[atoms.get_atomic_numbers()==8]=1.5
    cut[atoms.get_atomic_numbers()==14]=1.05
    cut[atoms.get_atomic_numbers()==1]=0.01

    nl=ase.neighborlist.NeighborList(cut,self_interaction=False)
    nl.update(atoms)

    

    pairs=[]
    counts=np.zeros(numAtoms)
    hClose=[]
    badO=[]


    #@TODO maybe just make a global bad list, distance can do as well.
    #first run through and make a list of all O atoms that have H that are too close
    for i in range(numAtoms):
        cur=atoms[i]
        curtype=cur.symbol
        
        if curtype == 'O':
            indices, offsets = nl.get_neighbors(i)
            hn=[]
            zpos=cur.position[2]
            #find all the H neighbors
            for n in indices:
                nei=atoms[n]
                neitype=nei.symbol
                if neitype =='H':
                    hn.append(n)
                    
            #check if too close
            if check_for_h(atoms,i,hn) == True:
                #print(f"atom {i+1} has H too close")
                hClose.append(i)
                badO.append(i)
                continue
            
            if position_check(zpos, zlowrange, zhighrange):
                badO.append(i)
                continue
        

                
    for i in range(numAtoms):
        cur=atoms[i]
        curtype=cur.symbol
        cpos=cur.position
        
        
        
        dprint(i,f'Debug - zpos:{cpos[2]}')
        
        #if this oxygen is a bad then skip
        if i in badO:
            continue
        
        if curtype == 'O':
            indices, offsets = nl.get_neighbors(i)
            
            #print(str(i+1)+ ' ' + str(indices))
            on=[]
            sin=[]
            hn=[]
            #create the list of good oxygen and Si neighbors to run through
            for n in indices:
                #if this oxygen is a bad then skip
                if n in badO:
                    continue
        
                nei=atoms[n]
                neitype=nei.symbol
                if neitype == 'O':
                    on.append(n)
                elif neitype =='Si':
                    sin.append(n)
                elif neitype =='H':
                    hn.append(n)
            
            
            dprint(i,f"debug - {on}")
            #run through the oxygen neighbors
            for n in on:
                nei=atoms[n]
                npos=nei.position
                
                dprint(i,f"debug testing {n}")
                    
                line=Line(cur.position,nei.position)
                
               
                #check if the pair vector is along the current direction we're picking pairs along
                if not perp_interface(cur,nei):
                    continue
                
                
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
            




    print(f"{len(hClose)} Oxygen have H that are too close")
    ids = [j+1 for j in range(numAtoms) if counts[j]>6]
    if len(ids)>0:
        print(f"Ids with greater than 6 pairs: {str(ids)}")
    ids = [j+1 for j in range(numAtoms) if counts[j]==6]
    if len(ids)>0:
        print(f"Ids with 6 pairs: {str(ids)}")
    ids = [j+1 for j in range(numAtoms) if counts[j]==5]
    if len(ids)>0:
        print(f"Number of ids with 5 pairs: {str(len(ids))}")

    pairfile=datapath+distDir+dfile[:-5]+"-pairlist.txt"
    if not os.path.exists(datapath+distDir):
        os.mkdir(datapath+distDir)
        
    shutil.copyfile(datapath+dfile, datapath+distDir+dfile)
    
    with open(pairfile,"w") as tf:
        for p in pairs:
            tf.write(f"{p[0]} {p[1]}\n")

    print(f"{len(pairs)} total pairs added to the file.")


def create_bond_file(L,datapath, file,bondfile):

    L.commands_string(f'''
        clear
        units         real
        dimension     3
        boundary    p p p
        atom_style  charge
        atom_modify map yes


        #atom_modify map array
        variable seed equal 12345
        variable NA equal 6.02e23

        timestep 0.5

        variable printevery equal 100
        variable restartevery equal 0
        variable datapath string "data/"


        variable massSi equal 28.0855 #Si
        variable massO equal 15.9991 #O
        variable massH equal  1.00784 #H
        

        read_data {datapath+file}
        
        mass         3 $(v_massH)
        mass         2 $(v_massO)
        mass         1 $(v_massSi)


        min_style quickmin
        
        pair_style	    reaxff potential/topcon.control
        pair_coeff	    * * potential/ffield_Nayir_SiO_2019.reax Si O H

        neighbor        2 bin
        neigh_modify    every 10 delay 0 check no
        
        thermo $(v_printevery)
        thermo_style custom step temp density press vol pe ke etotal #flush yes
        thermo_modify lost ignore
        
        region sim block EDGE EDGE EDGE EDGE EDGE EDGE
        
        fix r1 all qeq/reax 1 0.0 10.0 1e-6 reaxff
        compute c1 all property/atom x y z
        fix f1 all reaxff/bonds 1 {datapath}{bondfile}
        
        run 0
        ''')



def read_bonds(file):
    alist=[]
    with open(file,'r') as f:

        for line in f:
            #skip the first 7 lines for header
            ainfo=[]
            l=line.split()
            if l[0] == '#':
                continue
            i=0#the current item in the line we're on

            id=int(l[0])
            typ=int(l[1])
            nb=int(l[2])
            ainfo.append(id)#atom ID
            ainfo.append(typ)#atom Type
            ainfo.append(nb)#number of bonds
            
            i=3
            blist=[]
            #now make a list of bonded atom id's 
            for j in range(nb):
                blist.append(int(l[i]))
                i+=1
                
            ainfo.append(blist)
            alist.append(ainfo)
    
    df = pd.DataFrame(alist,columns=['id','type','nb','bonds'])
        
    return df
                
            
if __name__=='__main__':

    #current run defines
    debugatom=-1
    
    datapath="/home/agoga/documents/code/topcon-md/data/NEB/"

    
    distDirectory='perpPairs/'
    
    try:
        dfile=sys.argv[1]
    except:
        dfile=datapath+"Hcon-1500-0.data"
        
    
    dlist = ["Hcon-1500-0.data","Hcon-1500-110.data","Hcon-1500-220.data","Hcon-1500-330.data","Hcon-1500-440.data","Hcon-1500-550.data","Hcon-1500-695.data","Hcon-1500-880.data","Hcon-1500-990.data"]
    
    # for dfile in dlist:
    #     create_pair_list(datapath,dfile,distDirectory)
    

    
    plt.rcParams["figure.autolayout"] = True
    
    #Need two lammps instances so that when removing an atom and minimizing we don't increase time for final NEB image minimization
    L= lammps('mpi',["-log",f'{datapath}/CreateBonds.log'])
    file=dlist[0]
    bondfile=file[:-5]+".bonds"
    create_bond_file(L,datapath,file,bondfile)
    df=read_bonds(datapath+bondfile)
    print(df)

