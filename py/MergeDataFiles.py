from lammps import lammps
import sys
import os
import shutil
import analysis #md
import matplotlib
import numpy as np
from ipyfilechooser import FileChooser
from matplotlib import pyplot as plt

def findZDim(file):
    with open(file,'r') as f:
            lines = [line.rstrip() for line in f]
            for l in lines:
                s=l.split(' ')
                if(s[-1] == 'zhi'):
                    return [float(s[0]),float(s[1])]
                

def mergeDataFiles(dfiles):
    zdims = []
    zextent=0
    buffer = 1
    fileOG = dfiles[0]
    
    
    for f in dfiles:
        zdims.append(findZDim(f))
        
    print(zdims)          
    for z in zdims:
        zextent = zextent + (z[1]-z[0]) + buffer
    

    print(zextent)

    #finial z dimensions
    zf_lo = zdims[0][0]
    zf_hi = zf_lo + zextent


##LAMMPS SCRIPT
    L = lammps('mpi')
    L.commands_string(f'''
        shell cd topcon/
        clear
        units         real
        dimension     3
        boundary    p p p
        atom_style  charge

        #atom_modify map array
        variable seed equal 12345
        variable NA equal 6.02e23

        variable dt equal 1
        variable latticeConst equal 5.43

        #timestep of 0.5 femptoseconds
        variable printevery equal 100
        variable restartevery equal 0#500000
        variable datapath string "data/"


        variable massSi equal 28.0855 #Si
        variable massO equal 15.9991 #O
        variable massH equal  1.00784 #H
        
        ''')
    
    L.commands_string(f'''
        read_data {fileOG}
        
        change_box all z final {zf_lo} {zf_hi}
                    ''')
    
    shift=-zf_lo
    for i in range(len(dfiles)):#skip 1st

        cz=zdims[i]
        zsize=cz[1]-cz[0]
        if i == 0:
            shift = shift + zsize + buffer
            continue
        
        f= dfiles[i]
        L.command(f'read_data {f} add append shift 0 0 {shift}')

        shift = shift + zsize + buffer
        
    L.commands_string(f'''
                      
        mass         1 $(v_massH)
        mass         2 $(v_massO)
        mass         3 $(v_massSi)

        pair_style	    reaxff potential/topcon.control
        pair_coeff	    * * potential/ffield_Nayir_SiO_2019.reax H O Si

        neighbor        2 bin
        neigh_modify    every 10 delay 0 check no

        
        thermo $(v_printevery)
        thermo_style custom step temp density vol pe ke etotal #flush yes
        
        
        fix r1 all qeq/reax 1 0.0 10.0 1e-6 reaxff
        
        min_style hftn
        # minimize 1.0e-8 1.0e-8 100 100
                      ''')
    
    L.command(f'write_data py/outdata.data')
    
    
if __name__ == "__main__":
    
    cwd=os.getcwd()
    cwd='/home/agoga/topcon'
    # fc = FileChooser(cwd)
    # display(fc)

    folder='/data/aSiOxSlices15/'
    f=cwd+folder
    folderpath=os.path.join(cwd,f)

    files=['a-SiOx_1-1.data','a-SiOx_1-3.data','a-SiOx_1-5.data','a-SiOx_1-6.data','a-SiOx_1-8.data']#

    folders=[]

    for f in files:
        #folders.append('output-farm/with-v-without-h-SiO_1-5/'+f)
        folders.append(folderpath+f)
    mergeDataFiles(folders)