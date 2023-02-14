from lammps import lammps
import sys
import os
import shutil
import analysis #md
import matplotlib
import numpy as np
from mpi4py import MPI
from matplotlib import pyplot as plt

def findZDim(file):
    with open(file,'r') as f:
            lines = [line.rstrip() for line in f]
            for l in lines:
                s=l.split(' ')
                if(s[-1] == 'zhi'):
                    return [float(s[0]),float(s[1])]
                
def scaledFileName(file,a):
    id=file.index('.data')
    outfile = file[:id] + "_"+str(int(a*100)) + file[id:]
    return outfile
#reduces the entire system size by some amounts
def lmpReduceSystemSize(file,a):
    ##LAMMPS SCRIPT
    L = lammps('mpi')
    me = MPI.COMM_WORLD.Get_rank()
    nprocs = MPI.COMM_WORLD.Get_size()
    
    zdim=findZDim(file)
    zlen=zdim[1]-zdim[0]
    
    id=file.index('.data')
    outfile = scaledFileName(file,a)
    
    #print("Proc %d out of %d procs has" % (me,nprocs),L)
    L.commands_string(f'''
        shell cd topcon/
        clear
        units         real
        dimension     3
        boundary    p p f
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
        
        read_data {file}
        
        change_box all z final 0 {zlen*a}
        
        mass         1 $(v_massH)
        mass         2 $(v_massO)
        mass         3 $(v_massSi)

        pair_style	    reaxff potential/topcon.control
        pair_coeff	    * * potential/ffield_Nayir_SiO_2019.reax H O Si

        neighbor        2 bin
        neigh_modify    every 10 delay 0 check no
        
        thermo $(v_printevery)
        thermo_style custom step temp density vol pe ke etotal #flush yes
        thermo_modify lost ignore
        
        dump d1 all custom 1 py/CreateSiOx.dump id type q x y z ix iy iz mass element vx vy vz
        dump_modify d1 element H O Si

        fix r1 all qeq/reax 1 0.0 10.0 1e-6 reaxff

        min_style hftn
        minimize 1.0e-8 1.0e-8 10 10
        
        reset_atom_ids sort yes
        
        write_data {outfile}
        
        ''')


def mergeDataFiles(dfiles,buffers=[]):
    zdims = []
    zextent=0
    default_buffer = .2
    fileOG = dfiles[0]
    
    if len(buffers) == 0:
        for i in range(len(dfiles)):
            buffers.append(default_buffer)
    
    for f in dfiles:
        zdims.append(findZDim(f))
        
    #print(zdims)          
    for i in range(len(zdims)):
        z=zdims[i]
        zextent = zextent + (z[1]-z[0]) + buffers[i]
    

    #print(zextent)

    #finial z dimensions
    zf_lo = zdims[0][0]
    zf_hi = zf_lo + zextent

    
    
##LAMMPS SCRIPT
    L = lammps('mpi')
    me = MPI.COMM_WORLD.Get_rank()
    nprocs = MPI.COMM_WORLD.Get_size()
    #print("Proc %d out of %d procs has" % (me,nprocs),L)
    L.commands_string(f'''
        shell cd topcon/
        clear
        units         real
        dimension     3
        boundary    p p f
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
            shift = shift + zsize + buffers[i]
            continue
        
        f= dfiles[i]
        L.command(f'read_data {f} add append shift 0 0 {shift}')

        shift = shift + zsize + buffers[i]
        
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
        thermo_modify lost ignore
        
        dump d1 all custom 1 py/CreateSiOx.dump id type q x y z ix iy iz mass element vx vy vz
        dump_modify d1 element H O Si
        
        
        
        fix r1 all qeq/reax 1 0.0 10.0 1e-6 reaxff
        
        fix zwalls all wall/reflect zlo EDGE zhi EDGE
        
        delete_atoms overlap 1 all all
        
        reset_atom_ids sort yes
        
        min_style hftn
        minimize 1.0e-8 1.0e-8 1000 1000
        
        
                      ''')
    
    L.command(f'write_data py/SiOGrad.data')
    
    
if __name__ == "__main__":
    
    cwd=os.getcwd()
    cwd='/home/agoga/documents/code/topcon-md'
    cwd='/home/agoga/topcon'
    # fc = FileChooser(cwd)
    # display(fc)

    folder='/data/aSiOxSlices15/'
    f=cwd+folder
    folderpath=os.path.join(cwd,f)

    
    
    offsets=[0,0,0,0,0]
    folders=[]

    flip = 1

    scale=.25
       
    if flip == 0:
        files=['a-SiOx_1-1.data','a-SiOx_1-3.data','a-SiOx_1-5.data','a-SiOx_1-6.data','a-SiOx_1-8.data']
    else:
        files=['a-SiOx_1-1.data','a-SiOx_1-3.data','a-SiOx_1-6.data','a-SiOx_1-8.data']

    for f in files:
        #folders.append('output-farm/with-v-without-h-SiO_1-5/'+f)
        if flip == 0:
            folders.append(folderpath+f)
        else:
            folders.append(scaledFileName(folderpath+f,scale))

    if flip == 0:
        for file in folders:
            lmpReduceSystemSize(file,scale)
    else:
        
        mergeDataFiles(folders)