#!/usr/bin/env python
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
    lstr=f'''minimize {etol} {etol} 100000 100000'''
    if L is not None:
        L.commands_string(lstr)
    else:
        return lstr
    

def create_lmp_file(file,out,dumpstep=0):
    lammps_str=f'''
        clear
        units         real
        dimension     3
        boundary    p p p
        atom_style  charge
        atom_modify map yes

        variable seed equal 12345
        variable NA equal 6.02e23
 

        variable printevery equal 100
        variable restartevery equal 0#500000
        variable datapath string "data/"
        timestep {dt}

        variable massSi equal 28.0855 #Si
        variable massO equal 15.9991 #O
        variable massH equal  1.00784 #H 
        print $(v_infile)
        print $(v_outfile)
        '''
    print(f"Loading file with path: {file}")
    if  file.endswith(".dump"):
        lammps_str+=f'''
        variable massH equal  1.00784 #H 
        print $(v_infile)
        print $(v_outfile)
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

    # lammps_str+=f'''
    #     '''
    elif file.endswith(".data") or file.endswith(".dat"):
        lammps_str+=f'''
        read_data {file}
                        '''

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
    lammps_str += NEB_min()
        
    lammps_str+=f'''
        group gSi type 1
        group gO type 2
        group gH type 3
        
        variable tot equal $(count(gSi)+count(gO)+count(gH))
        variable Htot equal count(gH)
        variable perctH equal round($(100*v_Htot/v_tot))
        
        print '~$(v_perctH)% Hydrogen'
        
        write_data {out} '''
    print(f"Saving file with path: {out}")
    return lammps_str

def extract_box(L):
    bbox=L.extract_box()
    return np.array([[bbox[0][0],bbox[1][0]],[bbox[0][1],bbox[1][1]],[bbox[0][2],bbox[1][2]]])

def find_atom_position(L,atomI):
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
    #Initialize and load the dump file
    
    lstr=create_lmp_file(file,out,dumpstep)

    L = lammps('mpi',cmdargs=["-var","infile",file,'-var',"outfile",out])
    L.commands_string(lstr)
    
    lstr=create_lmp_file(file,out,dumpstep)

    L = lammps('mpi',cmdargs=["-var","infile",file,'-var',"outfile",out])
    L.commands_string(lstr)
    
    recenter_sim(L)
    
    L.commands_string('''
                      write_data /home/agoga/documents/code/topcon-md/data/neb/centered_Hcon-1500-695.dat
                      ''')
    

def prep_data(file,dumpstep,outfolder):
    me = MPI.COMM_WORLD.Get_rank()
    nprocs = MPI.COMM_WORLD.Get_size()
    
    outfile=file.removesuffix('.dat').removesuffix('.data').removesuffix('.dump')+".data"
    print(outfile)
    #L = lammps('mpi',["-var",f"infile {file}",'-var',f"outfile {outfile}"])
    create_dat(file,outfile,dumpstep)

         
    outfile=file.removesuffix('.dat').removesuffix('.data').removesuffix('.dump')+".data"
    print(outfile)
    #L = lammps('mpi',["-var",f"infile {file}",'-var',f"outfile {outfile}"])
    create_dat(file,outfile,dumpstep)

    return


    


    
if __name__ == "__main__":
    
    cwd=os.getcwd()

    folder='/data' #/pinhole-dump-files/'
    f=cwd+folder
    folderpath=os.path.join(cwd,f)

    # flist=["Hcon-1500-110.dump","Hcon-1500-220.dump","Hcon-1500-330.dump","Hcon-1500-440.dump","Hcon-1500-550.dump","Hcon-1500-695.dump","Hcon-1500-880.dump","Hcon-1500-990.dump"]
    # flist=["1.6-381.dat","1.7-276.dat","1.8-280.dat"]
    folderpath="/home/agoga/documents/code/topcon-md/data/pinhole-dump-files/"
    outfolder="/home/agoga/documents/code/topcon-md/data/NEB/"
    f="Hcon-1500-695.dump"
    dumpstep=1510053
    filepath=os.path.join(folderpath,f)
    nebFiles =prep_data(filepath,dumpstep,outfolder)

    # outfolder="/home/agoga/documents/code/topcon-md/data/NEB/"
    # #filepath=os.path.join(folderpath,file)
    # #prep_data(filepath,dumpstep,outfolder)
    # for f in flist:
    #     filepath=os.path.join(folderpath,f)
    #     nebFiles =prep_data(filepath,dumpstep,outfolder)
        
    MPI.Finalize()
    exit()