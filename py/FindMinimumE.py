 
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ovito.io import import_file, export_file
from ovito.data import *
from ovito.modifiers import *
from ovito.vis import Viewport
import matplotlib.gridspec as gridspec


##Taken from Icemtel's answer 
###https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib/56699813#56699813 
import matplotlib as mpl
class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))


a=5.43
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


def init_dump(L,file,out):
    #Initialize and load the dump file
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
        variable latticeConst equal {a}

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

        lattice none 1.0
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

        log FindMinimumE.log
        
        fix r1 all qeq/reax 1 0.0 10.0 1e-6 reaxff
        compute c1 all property/atom x y z

        write_data {out}
        ''')
    
def init_dat(L,file,out):
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


        #timestep of 0.5 femptoseconds
        variable printevery equal 100
        variable restartevery equal 0#500000
        variable datapath string "data/"


        variable massSi equal 28.0855 #Si
        variable massO equal 15.9991 #O
        variable massH equal  1.00784 #H
        

        read_data {file}
        
        mass         3 $(v_massH)
        mass         2 $(v_massO)
        mass         1 $(v_massSi)


        min_style quickmin
        
        pair_style	    reaxff potential/topcon.control
        pair_coeff	    * * potential/ffield_Nayir_SiO_2019.reax H O Si

        neighbor        2 bin
        neigh_modify    every 10 delay 0 check no
        
        thermo $(v_printevery)
        thermo_style custom step temp density vol pe ke etotal #flush yes
        thermo_modify lost ignore
        
        dump d1 all custom 1 py/CreateSiOx.dump id type q x y z ix iy iz mass element vx vy vz
        dump_modify d1 element H O Si

        log FindMinimumE.log
        
        ''')

def wigglewiggle(file,atom):
 ##LAMMPS SCRIPT
    L = lammps('mpi')
    L2 = lammps('mpi')
    L3= lammps('mpi')
    me = MPI.COMM_WORLD.Get_rank()
    nprocs = MPI.COMM_WORLD.Get_size()

    full='py/findMinFull.data'
    out='py/findMinInitial.data'
    init_dump(L2,file,full)#do this to get around reaxff issues with deleting atoms and writing data
    
    init_dat(L,full,out)
    
    
    #now delete atoms 
    L.commands_string(f'''
        variable xi equal x[{atom}]
        variable yi equal y[{atom}]
        variable zi equal z[{atom}]
        ''')
    
    bbox= L.extract_box()
    bbox=[[bbox[0][0],bbox[1][0]],[bbox[0][1],bbox[1][1]],[bbox[0][2],bbox[1][2]]]
    
    xi = L.extract_variable('xi')
    yi = L.extract_variable('yi')
    zi = L.extract_variable('zi')
    
    xzhalfwidth = 15.1
    yhwidth=1.5
    step = .5
    buff=1
    
    xrange = [max(xi-buff*xzhalfwidth,  bbox[0][0]),    min(xi+buff*xzhalfwidth,    bbox[0][1])]
    yrange = [max(yi-buff*yhwidth,      bbox[1][0]),    min(yi+buff*yhwidth,        bbox[1][1])]
    zrange = [max(zi-buff*xzhalfwidth,  bbox[2][0]),    min(zi+buff*xzhalfwidth,    bbox[2][1])]

    L.commands_string(f'''
        
        region sim block EDGE EDGE EDGE EDGE EDGE EDGE
        region ins block {xrange[0]} {xrange[1]} {yrange[0]} {yrange[1]} {zrange[0]} {zrange[1]} units box 
        region outs intersect 2 sim ins side out
        delete_atoms region outs compress no
        
        change_box all x final {xrange[0]} {xrange[1]} y final {yrange[0]} {yrange[1]} z final {zrange[0]} {zrange[1]} units box 
        
        
        fix r1 all qeq/reax 1 0.0 10.0 1e-6 reaxff
        compute c1 all property/atom x y z
        
        run 0

        write_data {out}
                      ''')
    
    
    try:
        pipeline = import_file(out)
    except Exception as e:
        print(e)
    pipeline.modifiers.append(ExpressionSelectionModifier(expression = f'ParticleIdentifier=={atom}'))
    data=pipeline.compute()
    
    pipeline.add_to_scene()
    vp = Viewport()
    vp.type = Viewport.Type.Front
    #vp.camera_pos = (0, 100, 0)
    vp.zoom_all()
    #vp.camera_dir = (2, 3, -3)
    #vp.fov = math.radians(60.0)
    
    ovitoFig="py/ovitoFig.png"
    vp.render_image(size=(800,600), filename=ovitoFig)
    

    
    Ei = L.extract_compute('thermo_pe',0,0)
    Ef=0
    


    xlist=np.arange(-xzhalfwidth,xzhalfwidth,step)
    zlist=np.arange(-xzhalfwidth,xzhalfwidth,step)
    
    
    xlen=len(xlist)
    zlen=len(zlist)
    
    elist=np.zeros([xlen,zlen])
    tot = xlen*zlen
    i=1
    for j in range(zlen):
        for k in range(xlen):
        
            y=0
            
            x=xlist[k]
            z=zlist[j]
            
            xf = xi + x
            yf = yi
            zf = zi + z
            # print(f"xf={xi}+{x}={xf}")
            # print(f"zf={zi}+{z}={zf}")
            print(f"Step {i}/{tot}")
            
            L.commands_string(f'''
                set atom {atom} x {xf} y {yf} z {zf}
                run 0
                ''')
            i+=1
            Ef = L.extract_compute('thermo_pe',0,0)
            dE=Ef-Ei
            elist[j,k]=dE

    plt.rcParams["figure.autolayout"] = True
    
    fig = plt.figure(figsize=(12,6))
    gs = gridspec.GridSpec(1, 2,width_ratios=[1,1.6])
                
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ovitoImage=plt.imread(ovitoFig)
    ax2.axis('off')
    ax2.imshow(ovitoImage,cmap='gray')
    
    
    #####
    #####
    #
    # norm = MidpointNormalize(vmin=np.min(elist), vmax=np.max(elist), midpoint=0)
    # cmap = 'RdBu_r' 
    # im=ax1.contourf(zlist,xlist,elist,20,cmap=cmap,norm=norm)
    

    im=ax1.contourf(zlist,xlist,elist,20,cmap='viridis')
    #####
    #####
    ax1.axis('scaled')
    ax1.set_xlabel('Δx(Å)')
    ax1.set_ylabel('Δz(Å)')
    
    
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.15)
    cbar=fig.colorbar(im,cax=cax,orientation='vertical')
    cbar.set_label('ΔE(kcal/mol)')
    
    ax1.set_title(f"Potential energy landscape around atom {atom}")
    plt.savefig(f"py/PES({atom}-0{int(10*step)}).png")
    
        
    



if __name__ == "__main__":
    
    cwd=os.getcwd()

    folder='/data/'
    f=cwd+folder
    folderpath=os.path.join(cwd,f)
    file="Hy2-1400.dump"
    filepath=os.path.join(folderpath,file)
    
    atomID=2661
    #Atom 4619 for middle of the c-Si 
    #Atom 4251 for right at interface
    #atoms: 1085(F), 332
    wigglewiggle(filepath,atomID)

