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
etol=1e-10
#conversion from kcal/mol to eV
conv=0.043361254529175


step = .5
buff=1



def NEB_min(L):
    L.commands_string(f'''minimize {etol} {etol} 100000 100000''')

def init_dump(L,file,out,dumpstep):
    #Initialize and load the dump file
    L.commands_string(f'''
        shell cd topcon/
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
        
        region sim block 0 1 0 1 0 1

        lattice diamond {a}

        create_box 3 sim

        read_dump {file} {dumpstep} x y z box yes add keep
        
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
        #compute c1 all property/atom x y z
        
        reset_atom_ids
        ''')
    
    NEB_min(L)
        
    L.commands_string(f'''
        group gSi type 2
        group gO type 1 
        group gH type 3
        
        variable tot equal $(count(gSi)+count(gO)+count(gH))
        variable Htot equal count(gH)
        variable perctH equal round($(100*v_Htot/v_tot))
        
        print '~$(v_perctH)% Hydrogen'
        
        write_data {out}
        ''')
    



def prep_data(file,dumpstep,outfolder):
    me = MPI.COMM_WORLD.Get_rank()
    nprocs = MPI.COMM_WORLD.Get_size()
    
    
    L = lammps('mpi')

    

    out=file[:-5]+".data"



    if file.endswith(".dump"):
        init_dump(L,file,out,dumpstep)

    else:
        print("File is not a .dump")
    
                
    return
    
if __name__ == "__main__":
    
    cwd=os.getcwd()

    folder='/data/'
    f=cwd+folder
    folderpath=os.path.join(cwd,f)

    

    file="pinhole-dump-files/Hcon-1500-440.dump"
    dumpstep=9

    file="SiOxNEB-NOH.dump"
    dumpstep=1
    
    file="aQ-SiO2.dump"
    dumpstep=21

    outfolder="/home/agoga/documents/code/topcon-md/data/"

    filepath=os.path.join(folderpath,file)
    nebFiles =prep_data(filepath,dumpstep,outfolder)
    MPI.Finalize()
    exit()