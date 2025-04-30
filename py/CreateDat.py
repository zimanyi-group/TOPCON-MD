#!/usr/bin/env python
"""
Author: Adam Goga
This script contains functions for setting up and running LAMMPS simulations that create minimized data files from already create samples.
This is needed to speed up NEB calculations later on in the pipeline. 
"""

from lammps import lammps
import sys
import os

import numpy as np
from mpi4py import MPI
from matplotlib import pyplot as plt
from random import gauss

from mpl_toolkits.axes_grid1 import make_axes_locatable
from ovito.io import import_file, export_file
from ovito.data import *
from ovito.modifiers import *
from ovito.vis import Viewport
import matplotlib.gridspec as gridspec
import matplotlib as mpl
 

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


a=5.43
dt=0.5
etol=7e-6
#conversion from kcal/mol to eV
conv=0.043361254529175


step = .5
buff=1



def NEB_min(L=None):
    """
    Add a minimization to current lammps run
    :param L - The lammps object
    :return The minimization command string
    """
    
    lstr=f'''minimize {etol} {etol} 100000 100000'''
    if L is not None:
        L.commands_string(lstr)
    else:
        return lstr
    

def create_lmp_file(file,out,dumpstep=0):
    """
    Create a LAMMPS input script file with specified parameters and write it to a specified output file.
    :param file - The input file
    :param out - The output file
    :param dumpstep - The dump step value (default is 0)
    :return LAMMPS input script string
    """
    lammps_str=f'''
        clear
        units         real
        dimension     3
        boundary    p p p
        atom_style  charge
        atom_modify map yes

        variable seed equal 12345
        variable NA equal 6.02e23
 

        variable printevery equal 10000
        variable restartevery equal 0#500000
        variable datapath string "data/"
        timestep {dt}

        variable massSi equal 28.0855 #Si
        variable massO equal 15.9991 #O
        variable massH equal  1.00784 #H 
        '''
    print(f"Loading file with path: {file}")
    if  file.endswith(".dump"):
        lammps_str+=f'''
        region sim block 0 1 0 1 0 1
        
        
        lattice diamond {a}

        create_box 3 sim

        read_dump {file} {dumpstep} x y z box yes add keep
        '''
    elif file.endswith(".data") or file.endswith(".dat"):
        lammps_str+=f'''
        read_data {file}
                        '''

    # # lammps_str+=f'''
    # #     '''
    # elif file.endswith(".data") or file.endswith(".dat"):
    #     lammps_str+=f'''
    #     read_data {file}
    #                     '''

    lammps_str+=f'''
        
        mass         3 $(v_massH)
        mass         2 $(v_massO)
        mass         1 $(v_massSi)

        lattice none 1.0
        min_style quickmin
        
        pair_style	    reaxff potential/topcon.control 
        pair_coeff	    * * potential/ffield_Nayir_SiO_2019.reax Si O H
        
        

        neighbor        2 bin
        neigh_modify    every 10 delay 0 check no
        
        thermo $(v_printevery)
        thermo_style custom step temp density press vol pe ke etotal #flush yes
        thermo_modify lost ignore

        log none
        
        fix r1 all qeq/reax 1 0.0 10.0 1e-6 reaxff

        
        reset_atom_ids
        
        '''
        
    lammps_str+=f'''
        group gSi type 1
        group gO type 2
        group gH type 3
        
        region wafer block EDGE EDGE EDGE EDGE 1.0 2.0 units lattice
        group CRYST region wafer



        group SILICON type 1
        group OXYGEN type 2
        group HYDROGEN type 3

        group cSi intersect CRYST SILICON

        group mobile subtract all cSi
        
        fix h1 mobile nvt temp 300 300 $(100.0 * dt)
        run  500000
        
        fix h1 mobile nvt temp 300 10 $(100.0 * dt)
        run  10000000
        
        '''
    lammps_str += NEB_min()
        
    lammps_str+=f'''
        
        
        
        variable tot equal $(count(gSi)+count(gO)+count(gH))
        variable Htot equal count(gH)
        variable perctH equal round($(100*v_Htot/v_tot))
        
        print '~$(v_perctH)% Hydrogen'
        
        write_data {out} '''
    
    write_file='/home/adam/code/topcon-md/lmp/CreateDat_current.lmp'
    with open(write_file,'a') as f:
        f.write(lammps_str)
    
    print(f"Saving file with path: {out}")
    return lammps_str

def extract_box(L):
    """
    Extract the bounding box coordinates from a running lammps script.
    :param L - The object from which the bounding box needs to be extracted.
    :return A numpy array containing the bounding box coordinates.
    """
    bbox=L.extract_box()
    return np.array([[bbox[0][0],bbox[1][0]],[bbox[0][1],bbox[1][1]],[bbox[0][2],bbox[1][2]]])

def find_atom_position(L,atomI):
    """
    Find the position of a specific atom in a LAMMPS simulation.
    :param L - The simulation box
    :param atomI - The index of the atom
    :return A tuple containing the x, y, and z coordinates of the atom
    """
    L.commands_string(f'''
        variable x{atomI} equal x[{atomI}]
        variable y{atomI} equal y[{atomI}]
        variable z{atomI} equal z[{atomI}]
        ''')
    
    x = L.extract_variable(f'x{atomI}')
    y = L.extract_variable(f'y{atomI}')
    z = L.extract_variable(f'z{atomI}')
    
    return (x,y,z)

def recenter_sim(L,atom_id=2370):
    """
    Recenter the simulation by moving the specified atom to the center of the box.
    :param L - The simulation box
    :param atom_id - The ID of the atom to recenter (default is 2370)
    :return The bounding box of the simulation.
    """
    
    r=find_atom_position(L,atom_id)
    bbox= extract_box(L)

    xhlen=abs(bbox[0][1]-bbox[0][0])/2
    yhlen=abs(bbox[1][1]-bbox[1][0])/2
    zhlen=abs(bbox[2][1]-bbox[2][0])/2
    # print(xhlen)
    # print(xhlen-r[0])
    
    L.commands_string(f'''
        
        #displace_atoms all move {xhlen-r[0]} {yhlen-r[1]} {zhlen-r[2]}
        displace_atoms all move {xhlen-r[0]} {yhlen-r[1]} 0
        run 0''')
    
    return bbox

def create_dat(file,out,dumpstep=0):
    """
    Create a LAMMPS instance using the provided input file and output file paths.
    :param file - The input file path.
    :param out - The output file path.
    :param dumpstep - The step at which to dump the data. Default is 0.
    """
    #Initialize and load the dump file
    
    lstr=create_lmp_file(file,out,dumpstep)

    L = lammps('mpi',cmdargs=["-var","infile",file,'-var',"outfile",out])
    L.commands_string(lstr)
    
    # recenter_sim(L)
    
    # L.commands_string('''
    #                   write_data /home/adam/code/topcon-md/data/neb/centered_Hcon-1500-695.dat
    #                   ''')
    

def prep_data(file,dumpstep,outfolder):
    """
    Prepare data for processing by removing unnecessary suffixes from the file name, creating a new output file, and calling a function to process the data.
    :param file - the input file to be processed
    :param dumpstep - the step for dumping data
    :param outfolder - the folder where the output file will be saved
    """
    me = MPI.COMM_WORLD.Get_rank()
    nprocs = MPI.COMM_WORLD.Get_size()
    
    outfile=file.removesuffix('.dat').removesuffix('.data').removesuffix('.dump')+".data"
    outfile=outfile.split('/')[-1]
    print(outfile)
    #L = lammps('mpi',["-var",f"infile {file}",'-var',f"outfile {outfile}"])
    create_dat(file,outfolder+outfile,dumpstep)

         
    # outfile=file.removesuffix('.dat').removesuffix('.data').removesuffix('.dump')+".data"
    # print(outfile)
    # #L = lammps('mpi',["-var",f"infile {file}",'-var',f"outfile {outfile}"])
    # create_dat(file,outfile,dumpstep)

    return


    


    
if __name__ == "__main__":
    
    cwd=os.getcwd()

    folder='/data' #/pinhole-dump-files/'
    f=cwd+folder
    folderpath=os.path.join(cwd,f)

    # flist=["Hcon-1500-110.dump","Hcon-1500-220.dump","Hcon-1500-330.dump","Hcon-1500-440.dump","Hcon-1500-550.dump","Hcon-1500-695.dump","Hcon-1500-880.dump","Hcon-1500-990.dump"]
    # flist=["1.6-381.dat","1.7-276.dat","1.8-280.dat"]
    # folderpath="/home/adam/code/topcon-md/data/pinhole-dump-files/"
    # outfolder="/home/adam/code/topcon-md/data/neb/"
    # f="Hcon-1500-695.dump"
    # dumpstep=1510053
    
    folderpath="/home/adam/code/topcon-md/data/neb/"
    outfolder="/home/adam/code/topcon-md/data/neb/minimized/"
    f="1.6-135.dat"
    filepath=os.path.join(folderpath,f)
    nebFiles =prep_data(filepath,0,outfolder)

    # outfolder="/home/agoga/documents/code/topcon-md/data/NEB/"
    # #filepath=os.path.join(folderpath,file)
    # #prep_data(filepath,dumpstep,outfolder)
    # for f in flist:
    #     filepath=os.path.join(folderpath,f)
    #     nebFiles =prep_data(filepath,dumpstep,outfolder)
        
    MPI.Finalize()
    exit()