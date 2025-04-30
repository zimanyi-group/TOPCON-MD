#!/usr/bin/env python
"""
Author: Adam Goga
This file contains functions which create pairlists for the NEB pipeline. It is the second step of the pipeline after creating and minimizing
your datafiles. These functions save a copy of the corresponding data file along with a pairlist file(.txt) into a folder which will be passed
to the NEB pipeline shell script.
"""

from mpi4py import MPI
import numpy as np
#from scipy import sparse
import matplotlib.pyplot as plt
#from skspatial.objects import Line, Sphere
import sys
from sympy import Point3D, Line3D, Plane
from sympy.abc import t

import os
import shutil
import pandas as pd
from pathlib import Path 


import string 

import NEBTools_setup as nt

me = MPI.COMM_WORLD.Get_rank()
numproc=MPI.COMM_WORLD.Get_size()


#used for perp/parallel to interface functions
planeDir=[0,0,1]

lowerletters=list(string.ascii_lowercase)

def dprint(id,str):
    if id == debugatom:
        print(str)

def perp_interface(cur, neigh):
    """
    Takes two atom ids and checks if their distance vector is perpendicular to the 'interface' and returns true or false
    :param cur - the current atom
    :param neigh - the neighboring atom
    :return True if the distance vector is perpendicular to the interface, False otherwise.
    """
 
    
    vecBuffer=20

    
    ang = nt.angle_between((cur-neigh),planeDir)

    if abs(ang) > vecBuffer:
        return False
    else:
        return True

def parallel_interface(cur, neigh):
    """
    Takes two atom ids and checks if their distance vector is parralel to the interface and returns true or false
    :param cur - the current atom
    :param neigh - the neighboring atom
    :return True if the distance vector is parallel to the interface, False otherwise.
    """
    
    vecBuffer=2

    #checking angle between pair sep vector and interface plane vector
    #so to be parallel the angle between needs to be roughly 90
    ang = 90 - nt.angle_between((cur-neigh),planeDir)

    if abs(ang) > vecBuffer:
        return False
    else:
        return True

def closer_pair(apos,bpos):
    """
    Takes two atom ids and checks which is closer to lower interface 
    :param apos - The position of point A in the form of (x, y, z)
    :param bpos - The position of point B in the form of (x, y, z)
    :return True if point A is closer to the reference point, False otherwise.
    """

    
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
    
def find_h2(datapath, dfile):
    """
    Find the number of hydrogen (H) atoms in the form of H2 in the system
    :param datapath - the path to the data file
    :param dfile - the data file to be processed
    :return None, just prints the list
    """
    

    
    filename=dfile.removesuffix('.dat').removesuffix('.data').removesuffix('.dump')

    #create_bond_file(datapath,dfile,bondfile)
    #atoms=read_bonds(datapath+'/scratchfolder/'+bondfile)
    (atoms, simbox) = nt.read_file_data_bonds(datapath,dfile)
    numAtoms=len(atoms.index)



    print(f'--------Running file: {dfile} with {numAtoms} atoms--------')

    pairs=[]

    counts=np.zeros(numAtoms+1)
    h2=[]


    Hatoms=atoms[atoms['type']=='H']

    
    #first run through and make a list of all O atoms that are bonded to a H or at too close to the interface
    for i, row in Hatoms.iterrows():
        curpos=row['pos']
        nindices = row['bonds']#[0] take only the indexes of the bonds not bond order
        hn=[]
        on=[]
        sin=[]
        zpos=curpos[2]
        
        h_nei=0
        o_nei=0
        si_nei=0
            
        #find all the H neighbors
        for ni in nindices:
            n=ni[0]
            bo=ni[1]
            neitype=atoms.at[n,'type']
            
            
            
            if neitype =='H':
                h_nei+=1
            elif neitype =='O':
                o_nei+=1
            elif neitype =='Si':
                si_nei+=1
                
        if h_nei>0:
            print(f"H atom {i}, z-{curpos[2]:0.2f} with neighbors; {h_nei}H, {o_nei}O,{si_nei}Si")
        
        
def create_bcH_to_anywhere_v2(datapath, dfile, distDir, writefile=False,split=1):
    """
    Find all BC H and move it to multiple locations to test which is the best final location to send it to. This version actually
    creates a number of locations which the neb pipeline will run through instead of just testing the pe quickly.

    :param datapath - the path to the data
    :param dfile - the data file
    :param distDir - the directory for the final distribution to be place in
    :param writefile - whether to write the file or not (default is False)
    :param split - how many files to split the pairs into, this is for running seperate calls on HPC clusters(default is 1).
    :return No return, but saves the original datafile and a text file with the final NEB settings to the distDir location 
            which will be read in by the NEB pipeline.
    """

    filename=dfile.removesuffix('.dat').removesuffix('.data').removesuffix('.dump')

    (atoms, simbox) = nt.read_file_data_bonds(datapath,dfile)
    numAtoms=len(atoms.index)

    zmin=18
    zmax=31


    print(f'--------Running file: {dfile} with {numAtoms} atoms--------')

    pairs=[]

    counts=np.zeros(numAtoms+1)


    Hatoms=atoms[atoms['type']=='H']

    # print(atoms.to_string())
    # print(atoms.columns)


    #run through all the Hydrogen atoms and find a place for them to move to
    for i, row in Hatoms.iterrows():
        if i != 3570:
            continue
        
        cpos=row['pos']
        nbonds = row['bonds']
        dprint(i,f'Debug - zpos:{cpos[2]}')
        
        
        zpos=cpos[2]
        
        if zpos < zmin or zpos > zmax:
            continue

        on=[]
        skip=False
        type_str=f"H Atom {i} has"
        o_count=0
        h_count=0
        si_count=0
        #check if it has 0 neighbors

        for ni in nbonds:
            n=ni[0]
            bo=ni[1]
            neitype=atoms.at[n,'type']
            
            if neitype =='O':
                o_count+=1
                skip=True
            elif neitype =='Si':
                si_count+=1
            elif neitype == 'H':
                h_count+=1
                skip=True
        
        if skip:
            continue
                    
        for ni in nbonds:
            n=ni[0]
            bo=ni[1]
            n_pos=atoms.at[n,'pos']
            nn_bonds=atoms.at[n,'bonds']
            
            for nni in nn_bonds:
                nn=nni[0]
                nn_type= atoms.at[nn,'type']
                nn_pos=atoms.at[nn,'pos']
                if nn_type == "O":
                    plane_vec=nt.pbc_vec_subtract(simbox,n_pos,nn_pos)
                    dplane=Plane(Point3D(nn_pos),normal_vector=plane_vec)
                    pt=dplane.arbitrary_point(t)
                    for theta in np.linspace(0,2*np.pi,num=4,endpoint=False):
                        
                        pairs.append([i,pt.subs({t:theta}).evalf()])

                
            

    npairs=np.array(pairs,dtype=object)

    if writefile and me==0:
        if not os.path.exists(datapath+distDir):
            os.mkdir(datapath+distDir)
        
        numpairs=len(pairs)

        pairlists=np.array_split(npairs,split)

        for i in range(split):
            curlen=len(pairlists[i])
            
            
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

def calc_bond_center(atoms,simbox,i,j):
    """
    Calculate the bond center location between two atoms.
    @param atoms - the atoms in the simulation
    @param simbox - the simulation box
    @param i - index of the first atom
    @param j - index of the second atom
    @return The midpoint between the two atoms
    """
    p1=atoms.at[i,'pos']
    p2=atoms.at[j,'pos']

    #get the midpoint of the neighboring vacancy
    midpt=p1+np.array(nt.pbc_vec_subtract(simbox,p1,p2))/2
    
    return midpt


def create_bcH_to_empty_space(datapath, dfile, distDir, writefile=False,split=1,min_dist=4):
    """
    Find all H in a BC and move it to nearby empty space with a min_dist to other atoms given

    :param datapath - the path to the data
    :param dfile - the data file
    :param distDir - the directory for the final distribution to be place in
    :param writefile - whether to write the file or not (default is False)
    :param split - how many files to split the pairs into, this is for running seperate calls on HPC clusters(default is 1).
    :param min_dist - min distance to all other atoms for the h to move into
    :return No return, but saves the original datafile and a text file with the final NEB settings to the distDir location 
            which will be read in by the NEB pipeline.
    """

    filename=dfile.removesuffix('.dat').removesuffix('.data').removesuffix('.dump')

    (atoms, simbox) = nt.read_file_data_bonds(datapath,dfile)
    numAtoms=len(atoms.index)



    print(f'--------Running file: {dfile} with {numAtoms} atoms--------')

    pairs=[]

    counts=np.zeros(numAtoms+1)


    Hatoms=atoms[atoms['type']=='H']

    file_path=datapath+dfile


    #run through all the Hydrogen atoms and find a place for them to move to
    for i, row in Hatoms.iterrows():
        # if i !=2548:
        #     continue
        # if i!= 361: #of 1.6-143
        cpos=row['pos']
        nbonds = row['bonds']
        dprint(i,f'Debug - zpos:{cpos[2]}')
        
        
        # zpos=cpos[2]
        
        # if zpos < zmin or zpos > zmax:
        #     continue
        
        o_count=0
        h_count=0
        si_count=0
        #check if it has 0 neighbors
        skip=False
        for ni in nbonds:
            n=ni[0]
            bo=ni[1]
            neitype=atoms.at[n,'type']
            
            
            if neitype =='O':
                o_count+=1
                skip=True
            elif neitype =='Si':
                si_count+=1
            elif neitype == 'H':
                h_count+=1
                skip=True
        if si_count == 2 or skip is False:
            continue
        
        print(f"----------H atom {i}---------")
        
        nearby_atoms=nt.find_nearby_atoms(file_path,i,7)
        pos_list=[]
        for n in nearby_atoms:
            pos_list.append([n,atoms.at[n,'pos']])
        
        empty_pos=nt.find_empty_space(simbox,cpos,pos_list,min_dist=min_dist,plot=True) #verify this is working correctly
        pairs.append([i,empty_pos])
        
        
    npairs=np.array(pairs,dtype=object)

    if writefile and me==0:
        if not os.path.exists(datapath+distDir):
            os.mkdir(datapath+distDir)
        
        numpairs=len(pairs)

        pairlists=np.array_split(npairs,split)

        for i in range(split):
            curlen=len(pairlists[i])
            
            
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
            print(f"{curlen} total pairs added to the file {datapath+distDir+pairname}.")
        
def create_bcH_to_anywhere_v1(datapath, dfile, distDir, writefile=False,split=1):
    """
    Find all BC H and move it to multiple locations to test which is the best final location to send it to.

    :param datapath - the path to the data
    :param dfile - the data file
    :param distDir - the directory for the final distribution to be place in
    :param writefile - whether to write the file or not (default is False)
    :param split - how many files to split the pairs into, this is for running seperate calls on HPC clusters(default is 1).
    :return No return, but saves the original datafile and a text file with the final NEB settings to the distDir location 
            which will be read in by the NEB pipeline.
    """

    filename=dfile.removesuffix('.dat').removesuffix('.data').removesuffix('.dump')

    (atoms, simbox) = nt.read_file_data_bonds(datapath,dfile)
    numAtoms=len(atoms.index)

    # zmin=18
    # zmax=31


    print(f'--------Running file: {dfile} with {numAtoms} atoms--------')

    pairs=[]

    counts=np.zeros(numAtoms+1)


    Hatoms=atoms[atoms['type']=='H']

    file_path=datapath+dfile


    #run through all the Hydrogen atoms and find a place for them to move to
    for i, row in Hatoms.iterrows():
        # if i != 281:
        #     continue
        if i == 384:
            break
        cpos=row['pos']
        nbonds = row['bonds']
        dprint(i,f'Debug - zpos:{cpos[2]}')
        print(f"----------H atom {i}---------")
        
        # zpos=cpos[2]
        
        # if zpos < zmin or zpos > zmax:
        #     continue

        
        nearby_atoms=nt.find_nearby_atoms(file_path,i,3)
        
        test_locations=[]
        for n_i in nearby_atoms:
            if n_i == i:
                continue
            n_type=atoms.at[n_i,'type']
            n_b=atoms.at[n_i,'bonds']
            print(f"Testing atom {n_i}")
            #if it's a Si, check for a O vacancy
            if n_type == "Si":
                si_vac=check_si_for_vacancy(atoms,n_i)
                for vac in si_vac:
                    r1=atoms.at[vac[0],'pos']
                    r2=atoms.at[vac[1],'pos']
                    
                    midpt=r1+np.array(nt.pbc_vec_subtract(simbox,r1,r2))/2
                    # print(f'Si adding {midpt}')
                    test_locations.append(midpt)
            elif n_type == "O":
                o_pos=atoms.at[n_i,'pos']
                test_pts=nt.gen_points_sphere(o_pos,1,16)
                
                for tp in test_pts:
                    dist=nt.pbc_dist(simbox,cpos,tp)
                    # print(f"distance {dist}")
                    
                    #@TODO not arbitrary
                    if dist < 2.5:
                        
                        df=nt.apply_dist_from_pos(atoms,simbox,tp)
                        too_close=df[df['dist']<.2]
                        # print(df[df['dist']<3].to_string())
                        
                        if len(too_close)==0:
                            # print(f"O adding {tp}")
                            test_locations.append(tp)
                            
                        
        loc_pe=[]       
        for loc in test_locations:
            try:
                loc_pe.append(nt.test_location_pe(file_path,i,loc))
            except:
                print(f'bad fail')
        
        # loc_pe=nt.test_multi_location_pe(file_path,i,test_locations)
        loc_pe=np.array(loc_pe,dtype=object)
        min_pe_arg=np.argmin(loc_pe[:,1])
        min_pe_loc=loc_pe[min_pe_arg][0]

        pairs.append([i,min_pe_loc])
        
                    
                    

 


   

    npairs=np.array(pairs,dtype=object)

    if writefile and me==0:
        if not os.path.exists(datapath+distDir):
            os.mkdir(datapath+distDir)
        
        numpairs=len(pairs)

        pairlists=np.array_split(npairs,split)

        for i in range(split):
            curlen=len(pairlists[i])
            
            
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
    

def create_H2_to_bc(datapath, dfile, distDir, writefile=False,split=1):
    """
    Currently only breaking a specific H2 to a specific BC.

    :param datapath - the path to the data
    :param dfile - the data file
    :param distDir - the directory for the final distribution to be place in
    :param writefile - whether to write the file or not (default is False)
    :param split - how many files to split the pairs into, this is for running seperate calls on HPC clusters(default is 1).
    :return No return, but saves the original datafile and a text file with the final NEB settings to the distDir location 
            which will be read in by the NEB pipeline.
    """
    

    
    

    filename=dfile.removesuffix('.dat').removesuffix('.data').removesuffix('.dump')

    #create_bond_file(datapath,dfile,bondfile)
    #atoms=read_bonds(datapath+'/scratchfolder/'+bondfile)
    (atoms, simbox) = nt.read_file_data_bonds(datapath,dfile)
    numAtoms=len(atoms.index)



    print(f'--------Running file: {dfile} with {numAtoms} atoms--------')

    pairs=[]
    
    
    filename=dfile.removesuffix('.dat').removesuffix('.data').removesuffix('.dump')

    #create_bond_file(datapath,dfile,bondfile)
    #atoms=read_bonds(datapath+'/scratchfolder/'+bondfile)
    (atoms, simbox) = nt.read_file_data_bonds(datapath,dfile)
    numAtoms=len(atoms.index)



    print(f'--------Running file: {dfile} with {numAtoms} atoms--------')

    pairs=[]

    counts=np.zeros(numAtoms+1)
    h2=[]


    Hatoms=atoms[atoms['type']=='H']

    
    #first run through and make a list of all O atoms that are bonded to a H or at too close to the interface
    for i, row in Hatoms.iterrows():
        cpos=row['pos']
        nindices = row['bonds']#[0] take only the indexes of the bonds not bond order
        hn=[]
        on=[]
        sin=[]
        zpos=cpos[2]
        
        h_count=0
        o_count=0
        si_count=0
            
        #find all the H neighbors
        for ni in nindices:
            n=ni[0]
            bo=ni[1]
            neitype=atoms.at[n,'type']
            
            
            if neitype =='O':
                o_count+=1
            elif neitype =='Si':
                si_count+=1
            elif neitype == 'H':
                h_count+=1
                hn.append(n)
        if h_count > 0 and o_count == 0 and si_count == 0:
            if i not in h2:
                h2.append(i)
            for n in hn:
                if n not in h2:
                    h2.append(n)
                    
                    
        print(f"H atom {i}, z-{cpos[2]:0.2f} with neighbors; {h_count}H, {o_count}O,{si_count}Si")  
    
    # H_atom=2501
    # bc_pairs=[[3154,3158],[3158,3152]]
    bc_pairs=[]
    for h_atom in h2:
        nearby_atoms=nt.find_nearby_atoms(datapath+dfile,h_atom,5)
        
        for n in nearby_atoms:
            neitype=atoms.at[n,'type']
            
            if neitype == 'Si':
                si_bc_vac=check_si_for_vacancy(atoms,n)
                if len(si_bc_vac) > 0:
                    for si_bc in si_bc_vac:
                        bc_pairs.append(si_bc)
    
        for bc_pair in bc_pairs:
            a1=bc_pair[0]
            a2=bc_pair[1]
            p1=atoms.at[a1,'pos']
            p2=atoms.at[a2,'pos']

            #get the midpoint of the neighboring vacancy
            midpt=p1+np.array(nt.pbc_vec_subtract(simbox,p1,p2))/2
        
            pl=[h_atom,midpt]

            pairs.append(pl)
   
    npairs=np.array(pairs,dtype=object)

    write_pairs_to_file(npairs,datapath,dfile,distDir,writefile,split)
    
    return pairs



def create_interface_H_to_H2(datapath, dfile, distDir, writefile=False,split=1):
    """
    Find all H near the interface, place another H atom nearby and set the final location of the NEB of the interface H 
    to become an H2.

    :param datapath - the path to the data
    :param dfile - the data file
    :param distDir - the directory for the final distribution to be place in
    :param writefile - whether to write the file or not (default is False)
    :param split - how many files to split the pairs into, this is for running seperate calls on HPC clusters(default is 1).
    :return No return, but saves the original datafile and a text file with the final NEB settings to the distDir location 
            which will be read in by the NEB pipeline.
    """

    filename=dfile.removesuffix('.dat').removesuffix('.data').removesuffix('.dump')

    (atoms, simbox) = nt.read_file_data_bonds(datapath,dfile)
    numAtoms=len(atoms.index)

    zmin=18
    zmax=31


    print(f'--------Running file: {dfile} with {numAtoms} atoms--------')

    pairs=[]



    Hatoms=atoms[atoms['type']=='H']


    #run through all the Hydrogen atoms and find a place for them to move to
    for i, row in Hatoms.iterrows():
        
        
        cpos=row['pos']
        nbonds = row['bonds']
        dprint(i,f'Debug - zpos:{cpos[2]}')
        
        
        zpos=cpos[2]
        
        if (zpos > zmin):
            continue

        on=[]
        skip=False
        type_str=f"H Atom {i} has"
        o_count=0
        h_count=0
        si_count=0
        #check if it has 0 neighbors

        for ni in nbonds:
            n=ni[0]
            bo=ni[1]
            neitype=atoms.at[n,'type']
            
            
            
            if neitype =='O':
                o_count+=1
            elif neitype =='Si':
                si_count+=1
            elif neitype == 'H':
                h_count+=1
 
        print(f"H atom {i}, z-{cpos[2]:0.2f} with neighbors; {h_count}H, {o_count}O,{si_count}Si")  
    
        nearby_atoms=nt.find_nearby_atoms(datapath+dfile,i,7)
        pos_list=[]
        for n in nearby_atoms:
            pos_list.append([n,atoms.at[n,'pos']])
        
        empty_pos=nt.find_empty_space(simbox,cpos,pos_list,min_dist=2.5,plot=False) #verify this is working correctly
        
        #direction from current H to empty pos
        dist = nt.pbc_dist(simbox,empty_pos,cpos)
        dir = nt.pbc_vec_subtract(simbox,cpos,empty_pos)
        norm_dir = nt.pbc_vec_subtract(simbox,cpos,empty_pos,normalize=True)
        
        
        avg_h_bond_len=0.78
        def pts(pt):
            return f"[{pt[0]:0.3f},{pt[1]:0.3f},{pt[2]:0.3f}]"
        

        
        final_loc = nt.vec_addition(cpos,(dist - avg_h_bond_len)*norm_dir)
        final_loc = nt.pbc_position_correction(simbox,final_loc)
        print(f"pos - {pts(cpos)}, dist - {dist:0.2f} empty_pos - {pts(empty_pos)}, dir - {pts(dir)}, final - {pts(final_loc)}")
        pairs.append([i,final_loc,empty_pos])

    

    npairs=np.array(pairs,dtype=object)

    if writefile and me==0:
        
        if not os.path.exists(datapath+distDir):
            os.mkdir(datapath+distDir)
        
        numpairs=len(npairs)

        pairlists=np.array_split(npairs,split)

        for i in range(split):
            curlen=len(pairlists[i])
            
            
            presplit=""
            if split>1:
                presplit=lowerletters[i]
                
            pairname=presplit+filename+"-pairlist.txt"
            
            pairfile=datapath+distDir+pairname

            shutil.copyfile(datapath+dfile, datapath+distDir+presplit+dfile)
            
            with open(pairfile,"w") as tf:
                for p in pairlists[i]:
                    #tf.write(f"{p[0]} {p[1]}\n")#[0]} {p[1][1]} {p[1][2]}\n")
                    tf.write(f"{p[0]} {p[1][0]} {p[1][1]} {p[1][2]} {p[2][0]} {p[2][1]} {p[2][2]}\n")
            print(f"{curlen} total pairs added to the file {datapath+distDir+pairname}.")
    
    return pairs
    

def create_interface_H_to_H2_region(datapath, dfile, distDir, writefile=False,split=1):
    """
    Find all H near the interface, place the atom and the direction we want the atom to go in the pairs file which will be
    read by the NEB pipeline.

    :param datapath - the path to the data
    :param dfile - the data file
    :param distDir - the directory for the final distribution to be place in
    :param writefile - whether to write the file or not (default is False)
    :param split - how many files to split the pairs into, this is for running seperate calls on HPC clusters(default is 1).
    :return No return, but saves the original datafile and a text file with the final NEB settings to the distDir location 
            which will be read in by the NEB pipeline.
    """

    filename=dfile.removesuffix('.dat').removesuffix('.data').removesuffix('.dump')

    (atoms, simbox) = nt.read_file_data_bonds(datapath,dfile)
    numAtoms=len(atoms.index)

    ziface=18
    zmin=3


    print(f'--------Running file: {dfile} with {numAtoms} atoms--------')

    pairs=[]



    Hatoms=atoms[atoms['type']=='H']


    #run through all the Hydrogen atoms and find a place for them to move to
    for i, row in Hatoms.iterrows():
        
        
        cpos=row['pos']
        nbonds = row['bonds']
        dprint(i,f'Debug - zpos:{cpos[2]}')
        
        
        zpos=cpos[2]
        
        if (zpos > ziface) or (zpos < zmin):
            continue

        on=[]
        skip=False
        type_str=f"H Atom {i} has"
        o_count=0
        h_count=0
        si_count=0
        #check if it has 0 neighbors

        for ni in nbonds:
            n=ni[0]
            bo=ni[1]
            neitype=atoms.at[n,'type']
            
            
            
            if neitype =='O':
                o_count+=1
            elif neitype =='Si':
                si_count+=1
            elif neitype == 'H':
                h_count+=1
 
        if h_count>0:
            continue
        print(f"H atom {i}, z-{cpos[2]:0.2f} with neighbors; {h_count}H, {o_count}O,{si_count}Si")  
    
        #direction from current H to empty pos

        norm_dir = np.array([0,0,-1])
        final_loc = nt.vec_addition(cpos,(4*norm_dir))
        final_loc = nt.pbc_position_correction(simbox,final_loc)
        
        
        def pts(pt):
            return f"[{pt[0]:0.3f},{pt[1]:0.3f},{pt[2]:0.3f}]"
   
        print(f"pos - {pts(cpos)},  dir - {pts(norm_dir)}, final - {pts(final_loc)}")
        pairs.append([i,final_loc])
    

    npairs=np.array(pairs,dtype=object)

    if writefile and me==0:
        
        if not os.path.exists(datapath+distDir):
            os.mkdir(datapath+distDir)
        
        numpairs=len(npairs)

        pairlists=np.array_split(npairs,split)

        for i in range(split):
            curlen=len(pairlists[i])
            
            
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
            print(f"{curlen} total pairs added to the file {datapath+distDir+pairname}.")
    
    return pairs



def create_H_to_all_bc(datapath, dfile, distDir, writefile=False,split=1):
    """
    Create pairlist to move any H atoms with a neighboring O vacancy to the BC of the participating Si atoms

    :param datapath - the path to the data
    :param dfile - the data file
    :param distDir - the directory for the final distribution to be place in
    :param writefile - whether to write the file or not (default is False)
    :param split - how many files to split the pairs into, this is for running seperate calls on HPC clusters(default is 1).
    :return No return, but saves the original datafile and a text file with the final NEB settings to the distDir location 
            which will be read in by the NEB pipeline.
    """

    filename=dfile.removesuffix('.dat').removesuffix('.data').removesuffix('.dump')

    (atoms, simbox) = nt.read_file_data_bonds(datapath,dfile)
    numAtoms=len(atoms.index)

    zmin=14.5
    zmax=32


    print(f'--------Running file: {dfile} with {numAtoms} atoms--------')

    pairs=[]

    counts=np.zeros(numAtoms+1)


    Hatoms=atoms[atoms['type']=='H']

    # print(atoms.to_string())
    # print(atoms.columns)


    #run through all the Hydrogen atoms and find a place for them to move to
    for i, row in Hatoms.iterrows():
        # if i != 216:
        #     continue
        
        cpos=row['pos']
        nbonds = row['bonds']
        dprint(i,f'Debug - zpos:{cpos[2]}')
        
        
        zpos=cpos[2]
        
        # if (zpos > 17 and zpos < 31):
        #     continue

        on=[]
        skip=False
        type_str=f"H Atom {i} has"
        o_count=0
        h_count=0
        si_count=0
        #check if it has 0 neighbors

        for ni in nbonds:
            n=ni[0]
            bo=ni[1]
            neitype=atoms.at[n,'type']
            
            
            
            
            
            if neitype =='O':
                o_count+=1
                skip=True
            elif neitype =='Si':
                si_count+=1
            elif neitype == 'H':
                h_count+=1
                skip=True
                
        # if o_count >0:
        #     print(f"{type_str} {si_count}Si, {o_count}O, {h_count}H neighbors")
        
        
        #Only moving H atoms that are BC to two Si.
        # if skip or si_count!=2:
        #     continue
                
        neighvac=find_neighboring_sibc(atoms,i)
            
            
        print(neighvac)
        if len(neighvac) ==0:
            continue

        mindist=100
        minpos=None
        minvac=None
        minsep=None
        for vac in neighvac:
            p1=atoms.at[vac[0],'pos']
            p2=atoms.at[vac[1],'pos']

            #get the midpoint of the neighboring vacancy
            midpt=p1+np.array(nt.pbc_vec_subtract(simbox,p1,p2))/2
        
            pl=[i,midpt]
            print(f"-----Atom {i} -> Best vacancy: {minvac} with dist={mindist}, adding {pl}------")
            pairs.append(pl)
            
            
            # #get the seperation vector between the vacancy and the attempted positon
            # sep=nt.pbc_vec_subtract(simbox,midpt,cpos)
            
            # vacdist=np.linalg.norm(sep)
            # #print(f"       poslist for {vac}: {curpos} {p1} {p2}\n       mid:{midpt} dist from guess:{vacdist}")
            # if vacdist < mindist:
            #     minvac=vac
            #     minpos=midpt
            #     mindist=vacdist
            #     min1=p1
            #     min2=p2

        # pl=[i,minpos]
        # print(f"-----Atom {i} -> Best vacancy: {minvac} with dist={mindist}, adding {pl}------")
        # pairs.append(pl)
        
        
        
        
        #now check if the vacancy is on the 'get out of pinhole' path
        
        
        
        # ang=nt.angle_between_pts(simbox,fpos,minpos,curpos)[0][0]
        # if ang > 50:
        #     #print(f"-----FAILED Atom {i} with ang={ang}, pos: {curpos}------\n------Best vacancy: {minvac} with dist={mindist}-----")
        #     continue
        
        #pos=nt.find_bond_preference(datapath+dfile,i,minpos,minsep)
        


   

    npairs=np.array(pairs,dtype=object)

    if writefile and me==0:
        print('Here1')
        if not os.path.exists(datapath+distDir):
            os.mkdir(datapath+distDir)
        
        numpairs=len(pairs)

        pairlists=np.array_split(npairs,split)

        for i in range(split):
            curlen=len(pairlists[i])
            
            
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
            print(f"{curlen} total pairs added to the file {datapath+distDir+pairname}.")
    
    return pairs

def create_wtmd_h(datapath, dfile, distDir, writefile=False,split=1):
    """
    Temp test function to create pairlist for checking barriers I use in WTMD runs
    """

    filename=dfile.removesuffix('.dat').removesuffix('.data').removesuffix('.dump')

    (atoms, simbox) = nt.read_file_data_bonds(datapath,dfile)
    numAtoms=len(atoms.index)

    zmin=14.5
    zmax=32


    print(f'--------Running file: {dfile} with {numAtoms} atoms--------')

    pairs=[]

    counts=np.zeros(numAtoms+1)


    Hatoms=atoms[atoms['type']=='H']


    # neighvac=check_si_for_vacancy(atoms,165)
        
    neighvac=[[237,260],[262,251],[251,252],[238,237]]
    print(neighvac)

    h_atom=472
    minsep=None
    for vac in neighvac:
        p1=atoms.at[vac[0],'pos']
        p2=atoms.at[vac[1],'pos']

        #get the midpoint of the neighboring vacancy
        midpt=p1+np.array(nt.pbc_vec_subtract(simbox,p1,p2))/2
    
        pl=[h_atom,midpt]
        print(f"-----Atom {h_atom} to bc of {vac[0]} & {vac[1]}, adding {pl}------")
        pairs.append(pl)
        
        
    


   

    npairs=np.array(pairs,dtype=object)

    if writefile and me==0:
        print('Here1')
        if not os.path.exists(datapath+distDir):
            os.mkdir(datapath+distDir)
        
        numpairs=len(pairs)

        pairlists=np.array_split(npairs,split)

        for i in range(split):
            curlen=len(pairlists[i])
            
            
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
            print(f"{curlen} total pairs added to the file {datapath+distDir+pairname}.")
    
    return pairs

def create_all_zap_pair_list(datapath, dfile, distDir,writefile=False,split=1):
    """
    Create a list of oxygen atom pairs to plug into the NEB pipeline using the ZAP method.
    :param datapath - the path to the data
    :param dfile - the data file
    :param distDir - the directory for the final distribution to be place in
    :param writefile - whether to write the file or not (default is False)
    :param split - how many files to split the pairs into, this is for running seperate calls on HPC clusters(default is 1).
    :return No return, but saves the original datafile and a text file with the final NEB settings to the distDir location 
                which will be read in by the NEB pipeline.
    """
    
    SiRad=10
    

    zlowrange=[18,20]
    zhighrange=[28,30]
    
 
    SiOBondOrder=nt.SiOBondOrder

 
    zmin=19
    zmax=28
    filename=dfile.removesuffix('.dat').removesuffix('.data').removesuffix('.dump')

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
            # print(f"O atom - {i} is bonded to a Si AND H.")  
            numoh+=1
            
              
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


        pairlists=np.array_split(npairs,split)

        for i in range(split):
            curlen=len(pairlists[i])
            
            
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


def find_neighboring_sibc(atoms,oi):
    """
    Find neighboring Silicon atoms with a BC vacancies starting from a given atom.
    :param atoms - The atoms in the system
    :param oi - The index of the starting atom
    :return A list of neighboring Silicon atoms with Boron and Carbon vacancies
    """

    
    nindices=atoms.at[oi,'bonds']
    #look to see if this atom has any neighboring SI with vacancies

    si_bc_vac=[]
    badsi=[]
    for n in nindices:
        
        ni=n[0]
        neitype=atoms.at[ni,'type']
        if neitype=='Si':

            si_vac=check_si_for_vacancy(atoms,ni)
            
            for siv in si_vac:
                si_bc_vac.append(siv)
    
    return si_bc_vac

def check_si_for_vacancy(atoms,si_i):
    """
    Check if a silicon atom is part of a Si-Si vacancy and return the two IDs involved.
    :param atoms - the atoms in the system
    :param si_i - the index of the silicon atom to check
    :return A list of two IDs representing the Si-Si vacancy, if found.
    """
    si_bc_vac=[]
    sibonds=atoms.at[si_i,'bonds']
            
    badsi=[]  
    for b in sibonds:
         
        bi=b[0]
        btype=atoms.at[bi,'type']
        #if we're bonded with a H or an O then any other Si bonded to that H or O isn't a vacancy with us
        if btype=='H' or btype=='O':
            bb=atoms.at[bi,'bonds']
            for bn in bb:
                bni=bn[0]
                bntype=atoms.at[bni,'type']
                if bntype=='Si' and bni != si_i:
                    badsi.append(bni)
    
    for nnn in sibonds:
        nnni=nnn[0]
        nnntype=atoms.at[nnni,'type']
        if nnntype=='Si'and nnni not in badsi:
            si_bc_vac.append([si_i,nnni])
            
    return si_bc_vac

def find_neighboring_sibc_recursion(atoms,si):          
    """
    Find all neighboring silicon atoms that are part of a vacancy chain recursively starting from a given silicon atom.
    :param atoms - the atoms in the system
    :param si - the index of the silicon atom to start from
    :return a list of silicon atoms that are part of a vacancy chain
    """

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
    """
    Find neighboring SI with a BC vacancy starting from a given oxygen atom.
    :param atoms - the atom structure
    :param curatom - the current atom to start from
    :return a list of neighboring silicon atoms with BC vacancies
    """

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
         



def create_pinhole_zap_pair_list(datapath, dfile, distDir,pinholeCenter,writefile=False,split=1):
    #pinatoms=[3032,3041,3053,3087,3091,3041,3091,3105,2985,2984,3087,3101,2985,3396,2985,2984,2984,3530,3521,1535,3530,3521,3434,3434,3323,3529,3482,3410,3434,3434,3478,3482,3573,3581,3493,3921,4021,3933,4024,4029,4025,4139,4141,4139,4106,6089,3445,3101,3445,3437,3439,3445,3434,3437,3434,3437,3491,3487,3491,3410,3411,3410,3491,3487,3493,3494,3409,3411,3497,3493,3534,3546,3109,3162,5102,3163,3124,3162,3162,3163,5050,3029,3469,3564,3564,3370,3370,3635,3564,3634,5533,3649,5578]
    """
    Create a list of oxygen atom pairs to plug into the NEB pipeline using the ZAP method. This method focuses on pairs that are within
    a pinhole.
    :param datapath - the path to the data
    :param dfile - the data file
    :param distDir - the directory for the final distribution to be place in
    :param pinholeCenter - the locations of the center of the pinhole
    :param writefile - whether to write the file or not (default is False)
    :param split - how many files to split the pairs into, this is for running seperate calls on HPC clusters(default is 1).
    :return No return, but saves the original datafile and a text file with the final NEB settings to the distDir location 
            which will be read in by the NEB pipeline.
    """
    SiRad=10
    

    zlowrange=[18,20]
    zhighrange=[28,30]
    

    SiOBondOrder=nt.SiOBondOrder


    zmin=19
    zmax=28
    maxRad=12
    filename=dfile.removesuffix('.dat').removesuffix('.data').removesuffix('.dump')

    
    (atoms, simbox) = nt.read_file_data_bonds(datapath,dfile)
    numAtoms=len(atoms.index)

    si_si_bond=0
#temp
    Siatoms=atoms[atoms['type']=='Si']
    for i, row in Siatoms.iterrows():
        curpos=row['pos']
        nindices = row['bonds']#
        
        if curpos[2]<zmin or curpos[2]>zmax:
            continue
        
        o_yes=False
        si_yes=False
        for ni in nindices:
            n=ni[0]
            bo=ni[1]
            neitype=atoms.at[n,'type']
            
            
            if neitype =='O':
                o_yes=True
            if neitype =='Si':
                si_yes=True
                
        if o_yes and si_yes:
            si_si_bond+=1
        
    print(f'Num Si_Si bonds {si_si_bond}')
#temp testing

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
        

        curpos=row['pos']
        nindices = row['bonds']#[0] take only the indexes of the bonds not bond order
        
        sepVec=nt.pbc_vec_subtract(simbox,pinholeCenter,curpos)
        
        
        #distance from pinhole center
        dist=nt.pbc_dist(simbox,pinholeCenter,curpos)

        
        # sepVecN=sepVec/np.linalg.norm(sepVec)
            
        # if dist > maxRad or not (curpos[2]>17 and curpos[2]<30):
        #     badO.append(i)
        #     continue
            
        
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
  
        #if there are any H neighbors
        if len(hn) > 0:
            #print(f"atom {i+1} has H too close")
            hClose.append(i)
            badO.append(i)
            continue

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
                
                p1=(i,nei)
                p2=(nei,i)
                
                # if i not in pinatoms and nei not in pinatoms:
                #     continue
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

    npairs=np.array(pairs)


    if writefile and me==0:
        
        if not os.path.exists(datapath+distDir):
            os.mkdir(datapath+distDir)


        pairlists=np.array_split(npairs,split)

        for i in range(split):
            curlen=len(pairlists[i])
            
            
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
    """
    Create a list of oxygen and hydrogen atoms to plug into the NEB pipeline using the ZAP method. This method focuses on OH complexes in our
    sample with the intention of moving both the oxygen and a hydrogen into a final location that was previously occupied by just a oxygen
    :param datapath - the path to the data
    :param dfile - the data file
    :param distDir - the directory for the final distribution to be place in
    :param pinholeCenter - the locations of the center of the pinhole
    :param writefile - whether to write the file or not (default is False)
    :param split - how many files to split the pairs into, this is for running seperate calls on HPC clusters(default is 1).
    :return No return, but saves the original datafile and a text file with the final NEB settings to the distDir location 
            which will be read in by the NEB pipeline.
    """
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

        pairlists=np.array_split(npairs,split)

        for i in range(split):
            curlen=len(pairlists[i])
            
            
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

def place_random_O(L,zlims,seed):
    """
    Create a random O atom within the limits given.
    :param L - The lammps simulation
    :param zlims - The limits for the z position
    :param seed - The random seed for the placement
    """
    L.commands_string(f''' 
                    region r_randO block EDGE EDGE EDGE EDGE {zlims[0]} {zlims[1]}
                    # #   create_atoms 2 random 1 12345 r_randO overlap 1.0 maxtry 1000
                    # group new_atom empty
                    fix fdep all deposit 1 2 1 {seed} region r_randO id max near 2
                    run 1
                      ''')

def create_interstitial_list(datapath, dfile, total_runs, distDir, pinholeCenter, writefile=False,split=1):
    """deprecated function"""
    # z_max=29
    # z_min=18

    # general_outfolder="/home/adam/code/topcon-md/output/"
    
    
    # for i in range(total_runs):
    #     filename=dfile.removesuffix('.dat').removesuffix('.data').removesuffix('.dump')


    # (atoms, simbox) = nt.read_file_data_bonds(datapath,dfile)
    # numAtoms=len(atoms.index)
    
    
    # print(f'--------Running file: {datapath+dfile} with {numAtoms} atoms--------')
    # atom_loc=[]
    
    
    # L1 = nt.get_lammps(f'{outfolder}/logs/PrepNEB-I.log')
    
    # fileIdent=f'{seed}'

    # reset1=outfolder+f'{fileIdent}-NEBI.dump'
    # reset2=outfolder+f'{fileIdent}-NEBF.dump'
    # nebI=outfolder+f'{fileIdent}-NEBI.data'
    # nebF=outfolder+f'{fileIdent}-NEBF.data'
    # full= outfolder+ f'{fileIdent}-Full.data'
    
    # PESimage=outfolder+f"PES({fileIdent}).png"
    # ovitoFig=outfolder+f"{fileIdent}-Ovito.png"
    
    # # selection=[atomI,atomF]
    
    
    # #initilize the data files 
    # if file.endswith(".dump"):
    #     LT = get_lammps(f'{outfolder}/logs/PrepNEB-LT.log')
    #     #do this first initialize to get around reaxff issues(charge stuff I think)
    #     init_dump(LT,file,dumpstep)
    #     LT.commands_string(f'''
    #         write_data {full}
    #         ''')
    #     #
    #     init_dat(L1,full)
    #     # init_dat(L2,full)
        
    # elif file.endswith(".data") or file.endswith(".dat"):
    #     init_dat(L1,file)
    #     # init_dat(L2,file)
    # else:
    #     print("File is not a .data or .dump")
    #     return
    
    # # place_random_O(L1,[bulk_low_z,bulk_high_z],seed)
    
    # atomI=L1.get_natoms()
    
    return 
    

def create_pinhole_pair_list_edge(datapath, dfile, distDir, pinholeCenter, writefile=False,split=1):
    """
    Create a list of oxygen atoms and final locations to plug into the NEB pipeline using the location method. This method 
    focuses on pairs that are near the edge of a pinhole.
    :param datapath - the path to the data
    :param dfile - the data file
    :param distDir - the directory for the final distribution to be place in
    :param pinholeCenter - the locations of the center of the pinhole
    :param writefile - whether to write the file or not (default is False)
    :param split - how many files to split the pairs into, this is for running seperate calls on HPC clusters(default is 1).
    :return No return, but saves the original datafile and a text file with the final NEB settings to the distDir location 
            which will be read in by the NEB pipeline.
    """
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
            minsep=None
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
                    minsep=sep
            
            
            
            
            
            #now check if the vacancy is on the 'get out of pinhole' path
            
            
            
            ang=nt.angle_between_pts(simbox,fpos,minpos,curpos)[0][0]
            if ang > 50:
                #print(f"-----FAILED Atom {i} with ang={ang}, pos: {curpos}------\n------Best vacancy: {minvac} with dist={mindist}-----")
                continue
            
            #pos=nt.find_bond_preference(datapath+dfile,i,minpos,minsep)
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
            curlen=len(pairlists[i])
            
            
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
    """
    Currently not working due to adding multi-jump functionality
    
    Create pairlist to move O atoms from the inner pinhole to the outer pinhole with multiple jumps

    :param datapath - the path to the data
    :param dfile - the data file
    :param distDir - the directory for the final distribution to be place in
    :param pinholeCenter - the locations of the center of the pinhole
    :param writefile - whether to write the file or not (default is False)
    :param split - how many files to split the pairs into, this is for running seperate calls on HPC clusters(default is 1).
    :return No return, but saves the original datafile and a text file with the final NEB settings to the distDir location 
            which will be read in by the NEB pipeline.
    """
    #
    
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

        
        sepVecN=sepVec/np.linalg.norm(sepVec)
            
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
            #nt.find_bond_preference
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
            curlen=len(pairlists[i])#double counts cause pairs
            
            
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

def create_all_H_neighbors_pair_list(datapath, dfile, distDir, writefile=False,split=1):
    """
    Create pairlist to move any H atoms within the pinhole to neighboring Si-Si BC

    :param datapath - the path to the data
    :param dfile - the data file
    :param distDir - the directory for the final distribution to be place in
    :param writefile - whether to write the file or not (default is False)
    :param split - how many files to split the pairs into, this is for running seperate calls on HPC clusters(default is 1).
    :return No return, but saves the original datafile and a text file with the final NEB settings to the distDir location 
            which will be read in by the NEB pipeline.
    """
    maxRad=12#from center of pinhole
    

    filename=dfile.removesuffix('.dat').removesuffix('.data').removesuffix('.dump')

    (atoms, simbox) = nt.read_file_data_bonds(datapath,dfile)
    numAtoms=len(atoms.index)
    

    print(f'--------Running file: {datapath+dfile} with {numAtoms} atoms--------')

    pairs=[]

    Oatoms=atoms[atoms['type']=='H']
    numadded=0
    #first run through and make a list of all O atoms that are bonded to a H or at too close to the interface
    for i, row in Oatoms.iterrows():
        # if numadded > 10:
        #     break
        curpos=row['pos']
        nindices = row['bonds']#[0] take only the indexes of the bonds not bond order

            
        
        neisibc=find_neighboring_sibc(atoms,i) #get all neighboring Si BC vacancies
        for sis in neisibc:
            a1=sis[0]
            a2=sis[1]
            p1=atoms.at[a1,'pos']
            p2=atoms.at[a2,'pos']
            midpt=nt.pbc_midpoint(simbox,p1,p2)

            sepv=nt.pbc_vec_subtract(simbox,p1,p2)
            pos=nt.find_bond_preference(simbox,datapath+dfile,i,midpt,sepv)
            pairs.append([i,pos,a1,a2])
            numadded+=1



    npairs=np.array(pairs,dtype=object)

    if writefile and me==0:
        
        if not os.path.exists(datapath+distDir):
            os.mkdir(datapath+distDir)
        
        numpairs=len(pairs)
        pairlists=np.array_split(npairs,split)

        for i in range(split):
            curlen=len(pairlists[i])#double counts cause pairs
            
            
            presplit=""
            if split>1:
                presplit=lowerletters[i]
                
            pairname=presplit+filename+"-pairlist.txt"
            
            pairfile=datapath+distDir+pairname

            shutil.copyfile(datapath+dfile, datapath+distDir+presplit+dfile)
            
            with open(pairfile,"w") as tf:
                for p in pairlists[i]:
                    tf.write(f"{p[0]} {p[1][0]} {p[1][1]} {p[1][2]} {p[2]} {p[3]}\n")
                    
                    
            print(f"{curlen} total pairs added to the file {pairname}.")
    
    return pairs


def create_all_O_neighbors_pair_list(datapath, dfile, distDir, pinholeCenter, writefile=False,split=1):
    """
    Create pairlist to move any O atoms within the pinhole to neighboring Si-Si BC

    :param datapath - the path to the data
    :param dfile - the data file
    :param distDir - the directory for the final distribution to be place in
    :param pinholeCenter - the locations of the center of the pinhole
    :param writefile - whether to write the file or not (default is False)
    :param split - how many files to split the pairs into, this is for running seperate calls on HPC clusters(default is 1).
    :return No return, but saves the original datafile and a text file with the final NEB settings to the distDir location 
            which will be read in by the NEB pipeline.
    """
    
    maxRad=12#from center of pinhole
    
    #how far to move the O atoms
    SiOBondOrder=nt.SiOBondOrder


    filename=dfile.removesuffix('.dat').removesuffix('.data').removesuffix('.dump')

    (atoms, simbox) = nt.read_file_data_bonds(datapath,dfile)
    numAtoms=len(atoms.index)
    

    print(f'--------Running file: {datapath+dfile} with {numAtoms} atoms--------')

    pairs=[]

    Oatoms=atoms[atoms['type']=='O']
    numadded=0
    #first run through and make a list of all O atoms that are bonded to a H or at too close to the interface
    for i, row in Oatoms.iterrows():
        # if numadded > 10:
        #     break
        curpos=row['pos']
        nindices = row['bonds']#[0] take only the indexes of the bonds not bond order
        
        sepVec=nt.pbc_vec_subtract(simbox,pinholeCenter,curpos)
        
        
        #distance from pinhole center
        dist=nt.pbc_dist(simbox,pinholeCenter,curpos)

        
        sepVecN=sepVec/np.linalg.norm(sepVec)
            
        if dist < maxRad and curpos[2]>17 and curpos[2]<30:
            neisibc=find_neighboring_sibc(atoms,i) #get all neighboring Si BC vacancies
            for sis in neisibc:
                a1=sis[0]
                a2=sis[1]
                p1=atoms.at[a1,'pos']
                p2=atoms.at[a2,'pos']
                midpt=nt.pbc_midpoint(simbox,p1,p2)

                sepv=nt.pbc_vec_subtract(simbox,p1,p2)
                pos=nt.find_bond_preference(simbox,datapath+dfile,i,midpt,sepv)
                pairs.append([i,pos,a1,a2])
                numadded+=1



    npairs=np.array(pairs,dtype=object)

    if writefile and me==0:
        
        if not os.path.exists(datapath+distDir):
            os.mkdir(datapath+distDir)
        
        numpairs=len(pairs)
        pairlists=np.array_split(npairs,split)

        for i in range(split):
            curlen=len(pairlists[i])#double counts cause pairs
            
            
            presplit=""
            if split>1:
                presplit=lowerletters[i]
                
            pairname=presplit+filename+"-pairlist.txt"
            
            pairfile=datapath+distDir+pairname

            shutil.copyfile(datapath+dfile, datapath+distDir+presplit+dfile)
            
            with open(pairfile,"w") as tf:
                for p in pairlists[i]:
                    tf.write(f"{p[0]} {p[1][0]} {p[1][1]} {p[1][2]} {p[2]} {p[3]}\n")
                    
                    
            print(f"{curlen} total pairs added to the file {pairname}.")
    
    return pairs


def find_nearby_O():
    """
    deprecated function
    """
    # distdf=nt.apply_dist_from_pos(atoms,simbox,fpos,"O")         
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
    return
    

def write_pairs_to_file(pairs,datapath, dfile, distDir, writefile=False,split=1):
    """
    Write pairs of atom data to a file to create a pairlist text file which will be further plugged into the NEB pipeline
    :param pairs - the pairs of data to write
    :param datapath - the path to the data
    :param dfile - the data file
    :param distDir - the directory for the output file
    :param writefile - flag to indicate whether to write the file
    :param split - the split value
    :return No return, but saves the original datafile and a text file with the final NEB settings to the distDir location 
            which will be read in by the NEB pipeline.
    """
    

    filename=dfile.removesuffix('.dat').removesuffix('.data').removesuffix('.dump')
    
    if writefile and me==0:
        print('Here1')
        if not os.path.exists(datapath+distDir):
            os.mkdir(datapath+distDir)
        
        numpairs=len(pairs)

        pairlists=np.array_split(pairs,split)

        for i in range(split):
            curlen=len(pairlists[i])
            
            
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
            print(f"{curlen} total pairs added to the file {datapath+distDir+pairname}.")

if __name__=='__main__':

    #current run defines
    debugatom=-1
    farmpath="/home/agoga/sandbox/topcon/data/neb/"
    datapath="/home/adam/code/topcon-md/data/neb/"
    # datapath="/home/adam/code/topcon-md/data/create_dat/"
    #datapath=farmpath
    
    
    #output directory

    distDirectory="pair_lists/bc_to_bc/"
    distDirectory="pair_lists/h2_to_bc/"
    distDirectory="pair_lists/wtmd/"
    fd=datapath  #+distDirectory

    if not os.path.exists(fd):
        os.makedirs(fd)
        

    
    #dlist = ["Hcon-1500-0.data","Hcon-1500-110.data","Hcon-1500-220.data","Hcon-1500-330.data","Hcon-1500-440.data","Hcon-1500-550.data","Hcon-1500-695.data","Hcon-1500-880.data","Hcon-1500-990.data"]

    i=1
    for d in Path(fd).glob('*.dat'):
        # if not str(d).endswith("1.6-381.dat"):
        if not str(d).endswith("jech_si_sio2_3.dat"):
            continue
        print(f"{str(i)}) {str(d)}")
        dfile=str(d).split('/')[-1]
        #create_pinhole_zap_pair_list(datapath,dfile,distDirectory,[27,27,20],True)
        print(f'{datapath} | {dfile} | {distDirectory}')
        
        create_wtmd_h(datapath,dfile,distDirectory,writefile=True)
        # create_interface_H_to_H2_region(datapath,dfile,distDirectory,writefile=True)
        #create_bcH_to_empty_space(datapath,dfile,distDirectory,writefile=True,min_dist=2)
        #create_H2_to_bc(datapath,dfile,distDirectory,writefile=True)
        
        
        #create_all_O_neighbors_pair_list(datapath,dfile,distDirectory,[27,27,20],True) #pinhole center [27,27,20],
        i+=1
     
     
     
     #Hcon-1500-695 - pinhole "center" ~ (21,10,22)

