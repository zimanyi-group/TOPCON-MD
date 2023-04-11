 
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
from matplotlib.colors import LightSource
from matplotlib import cbook
from matplotlib import cm

def fibonacci_sphere(r,samples=1000):

    points = []
    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append([x,y,z])

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


        min_style quickmin
        
        pair_style	    reaxff potential/topcon.control# safezone 1.6 mincap 100
        pair_coeff	    * * potential/ffield_Nayir_SiO_2019.reax H O Si

        neighbor        2 bin
        neigh_modify    every 10 delay 0 check no
        
        thermo $(v_printevery)
        thermo_style custom step temp density vol pe ke etotal #flush yes
        thermo_modify lost ignore
        
        dump d1 all custom 1 py/CreateSiOx.dump id type q x y z ix iy iz mass element vx vy vz
        dump_modify d1 element H O Si

        fix r1 all qeq/reax 1 0.0 10.0 1e-6 reaxff
        compute c1 all property/atom x y z
        
        run 0
        
        #write_data py/findMinInitial.data
        
        #minimize 1.0e-5 1.0e-5 5000 5000
        
        write_data py/findMinInitial.data
        

        variable xi equal x[{atom}]
        variable yi equal y[{atom}]
        variable zi equal z[{atom}]
        
        ''')
    
    xi = L.extract_variable('xi')
    yi = L.extract_variable('yi')
    zi = L.extract_variable('zi')
    Ei = L.extract_compute('thermo_pe',0,0)
    
    points=[]
    
    # #Spherical point creation
    # radii=np.linspace(1.7,5,num=20)
    # numP=100
    # for r in radii:
    #     points.extend(fibonacci_sphere(r,numP))


    width = 4.1
    step = .3
    
    i=1
    Em=3000000
    Ef=0
    
    res=[]
    res.append(tuple((0,0,0,0)))

    xlist=np.arange(-width,width,step)
    zlist=np.arange(-width,width,step)
    
    
    xlen=len(xlist)
    zlen=len(zlist)
    
    elist=np.zeros([xlen,zlen])
    tot = xlen*zlen
    i=1
    for zi in range(zlen):
        for xi in range(xlen):
        
            y=0
            
            x=xlist[xi]
            z=zlist[zi]
            
            xf = xi + x
            yf = yi
            zf = zi + z
            print(f"Step {i}/{tot}")
            
            L.commands_string(f'''
                set atom {atom} x {xf} y {yf} z {zf}
                run 0
                ''')
            i+=1
            Ef = L.extract_compute('thermo_pe',0,0)
            dE=Ef-Ei
            elist[zi,xi]=dE

            #res.append(tuple((dE,x,y,z)))
            
 

    # ml = np.array(nl
    plt.contourf(zlist,xlist,elist,35,cmap='viridis')
    plt.axis('scaled')
    plt.xlabel('Δx(Å)')
    plt.ylabel('Δz(Å)')
    plt.colorbar()
    plt.savefig(f"py/heatMap({atom}-0{int(10*step)}).png")
    
    
    
    #     if dE < Em:
    #         #print(f"New Min: {dE}")
    #         Em=dE
    #         (xm,ym,zm)=(xf,yf,zf)

            
    # Em=Ef-Ei
    # print(f"Final values\nEf={Ef}\nEi={Ei}\nEm={Em}\nxm={xm}\nym={ym}\nzm={zm}")
    
    # L.commands_string(f'''
    #         set atom {atom} x {xm} y {ym} z {zm}
    #         #minimize 1.0e-5 1.0e-5 5000 5000
    #         write_data py/findMinFinal.data
    #         ''')
    
    # print(res)
    # fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    # ls = LightSource(270, 45)
    # nl = list(map(list, zip(*res)))
    # ml = np.array(nl)
    
    # pe=ml[0]
    # xs=ml[1]
    # ys=ml[2]
    # zs=ml[3]
    # print(pe)


    # my_col = cm.jet(pe/np.amin(pe))
    # surf = ax.plot_trisurf(xs, ys, zs, rstride=1, cstride=1, facecolors=my_col,
    #                linewidth=0, antialiased=False)
    # plt.show()
        
    
    
    

if __name__ == "__main__":
    
    cwd=os.getcwd()

    folder='/data/'
    f=cwd+folder
    folderpath=os.path.join(cwd,f)
    file="Hy2-1400.dump"
    filepath=os.path.join(folderpath,file)
    
    atomID=984
    #Atom 4619 for middle of the c-Si 
    #atoms: 1085(F), 332
    wigglewiggle(filepath,atomID)
