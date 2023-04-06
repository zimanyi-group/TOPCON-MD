 
from lammps import lammps
import sys
import os
import shutil
import analysis #md
import matplotlib
import numpy as np
from mpi4py import MPI
from matplotlib import pyplot as plt
from random import gauss
import math


def fibonacci_sphere(r,samples=1000):

    points = []
    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append([r*x, r*y, r*z])

    return points

def make_rand_vector(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]

#current low
#-274000.38135801896
#-102907.31548723357
#6.520912860299781
#23.780024633166196
#25.442427850042495

# Final values
# Ef=-386070.54745449964
# Ei=-274000.38135801896
# Em=-112070.16609648068
# xm=7.1409396250638375
# ym=23.23457008771165
# zm=24.896108758550668

def wigglewiggle(file,atom):
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
        boundary    p p p
        atom_style  charge
        atom_modify map yes


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
        
        region sim block 0 1 0 1 0 1

        lattice diamond $(v_latticeConst)

        create_box 3 sim

        read_dump {file} 10000 x y z box yes add keep
        
        
        mass         3 $(v_massH)
        mass         2 $(v_massO)
        mass         1 $(v_massSi)


        min_style fire
        
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
        
        minimize 1.0e-3 1.0e-3 5000 5000
        
        compute c1 all property/atom x y z
        run 0
        variable xi equal x[{atom}]
        variable yi equal y[{atom}]
        variable zi equal z[{atom}]
        
        print '$(v_xi)'
        
        ''')
    
    xi = L.extract_variable('xi')
    yi = L.extract_variable('yi')
    zi = L.extract_variable('zi')
    Ei = L.extract_compute('thermo_pe',0,0)
    
    points=[]
    points = fibonacci_sphere(1,100)
    points.extend(fibonacci_sphere(1.5,100))
    points.extend(fibonacci_sphere(2,100))
    
    i=1
    Em=3000000
    xm=6.520912860299781
    ym=23.780024633166196
    zm=25.442427850042495
    Ef=0
    for p in points:
        xf = xi + p[0]
        yf = yi + p[1]
        zf = zi + p[2]
        L.commands_string(f'''
            set atom {atom} x {xf} y {yf} z {zf}
            print {i}
            minimize 1.0e-5 1.0e-5 5000 5000
            ''')
        i+=1
        Ef = L.extract_compute('thermo_pe',0,0)
        dE=Ef-Ei
        if dE < Em:
            Em=dE
            (xm,ym,zm)=(xf,yf,zf)
            print('New min' + str(Em))
    Em=Ef-Ei
    print(f"Final values\nEf={Ef}\nEi={Ei}\nEm={Em}\nxm={xm}\nym={ym}\nzm={zm}")
    

if __name__ == "__main__":
    
    cwd=os.getcwd()

    folder='/data/'
    f=cwd+folder
    folderpath=os.path.join(cwd,f)
    file="Hy2-1400.dump"
    filepath=os.path.join(folderpath,file)
    
    wigglewiggle(filepath,1085)
