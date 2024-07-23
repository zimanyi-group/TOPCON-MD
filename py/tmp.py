#!/usr/bin/env python
from lammps import lammps
import sys
import os

import numpy as np
from mpi4py import MPI

def get_lammps(log):
    return lammps('mpi',["-log",log,'-screen','none'])

def init_dump(L,file,dumpstep):
    #Initialize and load the dump file
    L.commands_string(f'''
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
        timestep 0.5

        variable massSi equal 28.0855 #Si
        variable massO equal 15.9991 #O
        variable massH equal  1.00784 #H
        
        region sim block 0 1 0 1 0 1

        lattice diamond 5.43
        
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
        
        dump d1 all custom 1 /home/adam/code/topcon-md/sandbox/tmp.dump id type xs ys zs
        dump_modify d1 element Si O H
        
        fix r1 all qeq/reax 1 0.0 10.0 1e-6 reaxff
        compute c1 all property/atom x y z
        
        minimize 1e-7 1e-7 1000 1000
        
        ''')
    
    
    
if __name__ == "__main__":
    L=get_lammps("/home/adam/code/topcon-md/sandbox/tmp.log")
    init_dump(L,"/home/adam/code/topcon-md/data/1.6-551_pinhole.dump",1620)