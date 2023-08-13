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
from ovito.vis import TachyonRenderer
import matplotlib.gridspec as gridspec
import matplotlib as mpl
 

me = MPI.COMM_WORLD.Get_rank()
#dt=1
etol=sys.argv[3]
dt=sys.argv[4]
skipPES=int(sys.argv[5])

a=5.43
#conversion from kcal/mol to eV
conv=0.043361254529175

xzhalfwidth = 20.1
yhwidth=5.1
step = .5
buff=1

xlist=np.arange(-xzhalfwidth,xzhalfwidth,step)
zlist=np.arange(-xzhalfwidth,xzhalfwidth,step)

xlen=len(xlist)
zlen=len(zlist)


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

def find_atom_position(L,atomID):
    L.commands_string(f'''
        variable x{atomID} equal x[{atomID}]
        variable y{atomID} equal y[{atomID}]
        variable z{atomID} equal z[{atomID}]
        ''')
    
    x = L.extract_variable(f'x{atomID}')
    y = L.extract_variable(f'y{atomID}')
    z = L.extract_variable(f'z{atomID}')
    
    return (x,y,z)

def NEB_min(L):
    L.commands_string(f'''minimize {etol} {etol} 10000 10000''')

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
        
        fix r1 all qeq/reax 1 0.0 10.0 1e-6 reaxff
        compute c1 all property/atom x y z''')
    
def init_dat(L,file):

    L.commands_string(f'''
        clear
        units         real
        dimension     3
        boundary    p p p
        atom_style  charge
        atom_modify map yes


        #atom_modify map array
        variable seed equal 12345
        variable NA equal 6.02e23

        timestep {dt}

        variable printevery equal 100
        variable restartevery equal 0
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
        pair_coeff	    * * potential/ffield_Nayir_SiO_2019.reax Si O H

        neighbor        2 bin
        neigh_modify    every 10 delay 0 check no
        
        thermo $(v_printevery)
        thermo_style custom step temp density press vol pe ke etotal #flush yes
        thermo_modify lost ignore
        
        region sim block EDGE EDGE EDGE EDGE EDGE EDGE
        
        fix r1 all qeq/reax 1 0.0 10.0 1e-6 reaxff
        compute c1 all property/atom x y z

        ''')

def create_ovito_plot(infile,figureName,r,atomID,selection=None):
    yslabwidth=5#ang
    xzslabwidth=20
    try:
        x=r[0]
        y=r[1]
        z=r[2]
        pipeline = import_file(infile)
        if selection == None:
            selection=[atomID]

        l=len(selection)
        expr=""
        for i in range(l):
            s=selection[i]
            expr+=f'ParticleIdentifier=={s}'
            if i != l-1:
                expr+='||'
                
        pipeline.modifiers.append(ExpressionSelectionModifier(expression = expr))
        pipeline.modifiers.append(AssignColorModifier(color=(0, 1, 0)))
        #pipeline.modifiers.append(SliceModifier(normal=(1,0,0),distance=x,slab_width=xzslabwidth))
        pipeline.modifiers.append(SliceModifier(normal=(0,1,0),distance=y,slab_width=yslabwidth))  
        #pipeline.modifiers.append(SliceModifier(normal=(0,0,1),distance=z,slab_width=xzslabwidth))
        
        data=pipeline.compute()
        
        pipeline.add_to_scene()
        vp = Viewport()
        vp.type = Viewport.Type.Front
        vp.zoom_all()
        
        # for i in range(20):
        #     print(vp.camera_pos)
        # vp.__setattr__("camera_pos",(x,vp.camera_pos[1],z))
        # vp.camera_pos[0]=x
        # vp.camera_pos[2]=z
        
        vp.render_image(size=(600,600), filename=figureName,renderer=TachyonRenderer(ambient_occlusion=False, shadows=False))
    except Exception as e:
        print(e)
        
def reduce_sim_box(L,rpos):
    xi=rpos[0]
    yi=rpos[1]
    zi=rpos[2]
    bbox= L.extract_box()
    bbox=[[bbox[0][0],bbox[1][0]],[bbox[0][1],bbox[1][1]],[bbox[0][2],bbox[1][2]]]
    
    
    xrange = [max(xi-buff*xzhalfwidth,  bbox[0][0]),    min(xi+buff*xzhalfwidth,    bbox[0][1])]
    yrange = [max(yi-buff*xzhalfwidth,  bbox[1][0]),    min(yi+buff*xzhalfwidth,    bbox[1][1])]
    zrange = [max(zi-buff*xzhalfwidth,  bbox[2][0]),    min(zi+buff*xzhalfwidth,    bbox[2][1])]

    L.commands_string(f'''
        
        region ins block {xrange[0]} {xrange[1]} {yrange[0]} {yrange[1]} {zrange[0]} {zrange[1]} units box 
        region outs intersect 2 sim ins side out
        delete_atoms region outs compress no
        
        change_box all x final {xrange[0]} {xrange[1]} y final {yrange[0]} {yrange[1]} z final {zrange[0]} {zrange[1]} units box 
        
        run 0''')


def create_PES(L,atom):

    xi, yi, zi = find_atom_position(L,atom)
    ri=(xi,yi,zi)

    Ei = L.extract_compute('thermo_pe',0,0)*conv
    Ef=0

    
    elist=np.zeros([xlen,zlen])
    tot = xlen*zlen
    i=1
    
    if skipPES != 1:
        for j in range(zlen):
            for k in range(xlen):
            
                y=0
                
                x=xlist[k]
                z=zlist[j]
                
                xf = xi + x
                yf = yi
                zf = zi + z
                
                print(f"Step {i}/{tot}")
                
                L.commands_string(f'''
                    set atom {atom} x {xf} y {yf} z {zf}
                    run 0
                    ''')
                i+=1
                Ef = L.extract_compute('thermo_pe',0,0)*conv
                dE=Ef-Ei
                elist[j,k]=dE
            
    # #place the atom back where it came from!
    L.commands_string(f'''
        set atom {atom} x {xi} y {yi} z {zi}
        run 0
        ''')

    return [L,elist,ri]#returning the file names of the initial position and the final neb xyz file

def plot_PES(PESimage,markerPts,xlist,zlist,elist,title):
    #Plotting below
    fig,ax = plt.subplots(figsize=(6,6))
    
     # set maximum value of the PES to be twice the lowest E or 18eV, whichever is highest
    minE=np.min(elist)
    maxE=np.max(elist)
    ab=maxE
    
    
    # if abs(minE) >ab:
    #     ab=abs(minE)
    # max=12
    # if minE < -max:
    #     minE=-max
    # if maxE > max:
    #     maxE=max
        
    # elist[elist<-max]=-max
    # elist[elist>max]=max
    #minE=-maxE
    # maxE=2*abs(minE)
    # if maxE<18:
    #     maxE=18
    # elist[elist>maxE]=maxE
    
    
    # if finalLoc is not None:
    #     redXPts[1]=[finalLoc[0],finalLoc[1]]
    norm = MidpointNormalize(vmin=minE,vmax=maxE,midpoint=0)
    if maxE==0 and minE ==0:
        norm=None
        
    im=plt.contourf(zlist,xlist,elist,20,cmap='bwr',norm=norm)
    
    for m in markerPts:
        pts=m[1]
        plt.scatter(pts[0],pts[1],marker=m[0],c='g',)
        
    plt.grid('on',linewidth=0.25,linestyle='--')

    plt.axis('scaled')
    plt.xlabel('Δx(Å)')
    plt.ylabel('Δz(Å)',labelpad=0.05)
    
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar=fig.colorbar(im,cax=cax,orientation='vertical')
    cbar.set_label('ΔE(eV)')
    ax.set_title(title)
    #ax.set_xticklabels(np.arange(-math.floor(xzhalfwidth),math.floor(xzhalfwidth)+1,2))
    plt.savefig(PESimage)
    
    ###OLD working double PES ovito
    # fig = plt.figure(figsize=(12,6))
    # gs = gridspec.GridSpec(1, 2,width_ratios=[1,1.6])
    # ax1 = plt.subplot(gs[0])
    # ax2 = plt.subplot(gs[1])
    
    # #Load the atomistic view and put it in the second subplot
    # ovitoImage=plt.imread(ovitoFig)
    # ax2.axis('off')
    # ax2.imshow(ovitoImage,cmap='gray')
    
    
    # redXPts=np.transpose([[0,0],[rMin[0],rMin[1]]])
    
    
    # # if finalLoc is not None:
    # #     redXPts[1]=[finalLoc[0],finalLoc[1]]
        
    # im=ax1.contourf(zlist,xlist,elist,20,cmap='viridis')
    # ax1.scatter(redXPts[0],redXPts[1],marker='x',c='r')

    # ax1.axis('scaled')
    # ax1.set_xlabel('Δx(Å)')
    # ax1.set_ylabel('Δz(Å)')
    
    
    # divider = make_axes_locatable(ax1)
    # cax = divider.append_axes('right', size='5%', pad=0.15)
    # cbar=fig.colorbar(im,cax=cax,orientation='vertical')
    # cbar.set_label('ΔE(kcal/mol)')
    # ax1.set_title(f"Potential energy landscape around atom {atom}")
    # plt.savefig(PESimage)

def recenter_sim(L,r):
    
    bbox= L.extract_box()
    #bbox=[[bbox[0][0],bbox[1][0]],[bbox[0][1],bbox[1][1]],[bbox[0][2],bbox[1][2]]]
    
    xhlen=abs(bbox[1][0]-bbox[0][0])/2
    yhlen=abs(bbox[1][1]-bbox[0][1])/2
    zhlen=abs(bbox[1][2]-bbox[0][2])/2
    
    
    L.commands_string(f'''
        
        #displace_atoms all move {xhlen-r[0]} {yhlen-r[1]} {zhlen-r[2]}
        displace_atoms all move {xhlen-r[0]} {yhlen-r[1]} 0
        run 0''')
    
    return np.array([[bbox[0][0],bbox[1][0]],[bbox[0][1],bbox[1][1]],[bbox[0][2],bbox[1][2]]])

def prep_neb_swap(file,dumpstep,atomI,outfolder,atomF,plot):
    plt.rcParams["figure.autolayout"] = True
    
    #Need two lammps instances so that when removing an atom and minimizing we don't increase time for final NEB image minimization
    L1 = lammps('mpi',["-log",f'{outfolder}/logs/PrepNEB-I.log'])
    L2 = lammps('mpi',["-log",f'{outfolder}/logs/PrepNEB-F.log'])

    
    fileIdent=f'{atomI}'

    reset1=outfolder+f'{fileIdent}-NEBI.dump'
    reset2=outfolder+f'{fileIdent}-NEBF.dump'
    nebI=outfolder+f'{fileIdent}-NEBI.data'
    nebF=outfolder+f'{fileIdent}-NEBF.data'
    full= outfolder+ f'{fileIdent}-Full.data'
    
    PESimage=outfolder+f"PES({fileIdent}).png"
    ovitoFig=outfolder+f"{fileIdent}-Ovito.png"
    
    selection=[atomI,atomF]
    
    
    #initilize the data files 
    if file.endswith(".dump"):
        LT = lammps('mpi',["-log",f'{outfolder}/logs/PrepNEB-LT.log'])
        #do this first initialize to get around reaxff issues(charge stuff I think)
        init_dump(LT,file,dumpstep)
        LT.commands_string(f'''
            write_data {full}
            ''')
        #
        init_dat(L1,full)
        init_dat(L2,full)
        
    elif file.endswith(".data") or file.endswith(".dat"):
        init_dat(L1,file)
        init_dat(L2,file)
    else:
        print("File is not a .data or .dump")
        return
    
    
    ##### L1 - create the initial NEB data file
    ri = find_atom_position(L1,atomI)
    rf = find_atom_position(L1,atomF)
    icoord=ri#have to use the coordinates before recentering
    fcoord=rf
    
    bbox=recenter_sim(L1,ri)


    ri = find_atom_position(L1,atomI)
    NEB_min(L1)
    L1.commands_string(f'''
    write_data {nebI}
    ''')
    
    ri = find_atom_position(L1,atomI)
    
     
    if me == 0 and plot:
        #Now create ovito plot of atoms for future use
        create_ovito_plot(nebI,ovitoFig,ri,atomI,selection)
    
    ret = create_PES(L1,atomI)
    elist=ret[1]
    
    rf = find_atom_position(L1,atomF)
    ri = find_atom_position(L1,atomI)

    #delete the output file so that we can rewrite it without the atom
    try:
        if me == 0:
            os.remove(nebI)
    except:
        print("bad os fail - Proc %d out of %d procs" % (comm.Get_rank(),comm.Get_size()))
        return
    ###
    # Now we start deleting atoms
    # After deleting atoms we will save then load the data so that lammps resets all the charges do not have a net charge
    ###
    
    #delete the atom at the final location
    L1.commands_string(f'''
        group gFAtom id {atomF}
        delete_atoms group gFAtom compress no
        reset_timestep 0
        write_dump all atom {reset1}
    ''')
    L1f = lammps('mpi',["-log",f'{outfolder}/logs/PrepNEB-If.log'])
    init_dump(L1f,reset1,0)
    NEB_min(L1f)
    ri = find_atom_position(L1f,atomI)
    L1f.commands_string(f'''
        write_data {nebI}
        ''')
    
    #now print info for the final data file
    L1f.commands_string(f'''
        print "pcsv_iPos [{icoord[0]},{icoord[1]},{icoord[2]}]"
        print "pcsv_fPos [{fcoord[0]},{fcoord[1]},{fcoord[2]}]"
        print "pcsv_box [[{bbox[0][0]},{bbox[0][1]}],[{bbox[1][0]},{bbox[1][1]}],[{bbox[2][0]},{bbox[2][1]}]]"
        ''')
    
    
    
    
    
    
    #####L2 - create the final NEB data file
    ri = find_atom_position(L2,atomI)
    recenter_sim(L2,ri)
    
    ri = find_atom_position(L2,atomI)
    NEB_min(L2)

    
    rf2 = find_atom_position(L2,atomF)
    
    
    #delete the atom at the final location and reload using a dump file so that lammps resets charges
    L2.commands_string(f'''
        group gFAtom id {atomF}
        delete_atoms group gFAtom compress no
        reset_timestep 0
        write_dump all atom {reset2}
    ''')
    L2f = lammps('mpi',["-log",f'{outfolder}/logs/PrepNEB-Ff.log'])
    init_dump(L2f,reset2,0)
    
    
    xyz=outfolder+f'{fileIdent}-NEBFXYZ.data'
    #now create the lowest energy position data file for NEB.
    L2f.commands_string(f'''
                set atom {atomID} x {rf2[0]} y {rf2[1]} z {rf2[2]}
                
                run 0
                write_data {xyz}
    ''')
    
    NEB_min(L2f)
    
    rf2 = find_atom_position(L2f,atomID)
    

    plottitle=f"Potential energy landscape around atom {atomID}"
    redXPts=np.transpose([[0,0]])
    allPts=[['x',redXPts]]
    
    if skipPES != 1 and me==0:
        plot_PES(PESimage,allPts,xlist,zlist,elist,plottitle)
    
    
    ####Now clean up the dump file to be the correct format for NEB runs
    if me == 0:## ONLY RUN ON ONE PROCESS
        
        with open(nebF,'w+') as f:
            f.write(f"1 \n{atomID} {rf2[0]} {rf2[1]} {rf2[2]}")
    return
    

def prep_neb_into_iface(file,dumpstep,atomI,outfolder,plot):
    plt.rcParams["figure.autolayout"] = True
    
    #Need two lammps instances so that when removing an atom and minimizing we don't increase time for final NEB image minimization
    L1 = lammps('mpi',["-log",f'{outfolder}/logs/PrepNEB-iface-I.log'])


    
    fileIdent=f'{atomI}'

    nebI=outfolder+f'{fileIdent}-NEBI.data'
    nebF=outfolder+f'{fileIdent}-NEBF.data'
    full= outfolder+ f'{fileIdent}-Full.data'
    
    PESimage=outfolder+f"PES({fileIdent}).png"
    ovitoFig=outfolder+f"{fileIdent}-Ovito.png"
    
    
    
    #initilize the data files 
    if file.endswith(".dump"):
        LT = lammps('mpi',["-log",f'{outfolder}/logs/PrepNEB-LT.log'])
        #do this first initialize to get around reaxff issues(charge stuff I think)
        init_dump(LT,file,dumpstep)
        LT.commands_string(f'''
            write_data {full}
            ''')
        #
        init_dat(L1,full)

        
    elif file.endswith(".data") or file.endswith(".dat"):
        init_dat(L1,file)

    else:
        print("File is not a .data or .dump")
        return
    
    
    ##### L1 - create the initial NEB data file
    ri = find_atom_position(L1,atomI)

    icoord=ri#have to use the coordinates before recentering
    fcoord=rf
    
    bbox=recenter_sim(L1,ri)


    ri = find_atom_position(L1,atomI)
    NEB_min(L1)
    L1.commands_string(f'''
    write_data {nebI}
    ''')
    
    ri = find_atom_position(L1,atomI)
    
     
    if me == 0 and plot:
        #Now create ovito plot of atoms for future use
        create_ovito_plot(nebI,ovitoFig,ri,atomI,selection)
    
    ret = create_PES(L1,atomI)
    elist=ret[1]
    
    ri = find_atom_position(L1,atomI)

    #delete the output file so that we can rewrite it without the atom
    try:
        if me == 0:
            os.remove(nebI)
    except:
        print("bad os fail - Proc %d out of %d procs" % (comm.Get_rank(),comm.Get_size()))
        return
    ###
    # Now we start deleting atoms
    # After deleting atoms we will save then load the data so that lammps resets all the charges do not have a net charge
    ###
    

    NEB_min(L1)
    ri = find_atom_position(L1,atomI)
    L1.commands_string(f'''
        write_data {nebI}
        ''')
    
    #now print info for the final data file
    L1.commands_string(f'''
        print "pcsv_iPos [{icoord[0]},{icoord[1]},{icoord[2]}]"
        print "pcsv_fPos [{fcoord[0]},{fcoord[1]},{fcoord[2]}]"
        print "pcsv_box [[{bbox[0][0]},{bbox[0][1]}],[{bbox[1][0]},{bbox[1][1]}],[{bbox[2][0]},{bbox[2][1]}]]"
        ''')
    
    
    recenter_sim(L2,ri)
    
    ri = find_atom_position(L1,atomI)
    NEB_min(L1)

    
    rf = find_atom_position(L1,atomF)
    
    
    
    xyz=outfolder+f'{fileIdent}-NEBFXYZ.data'
    #now create the lowest energy position data file for NEB.
    L1.commands_string(f'''
                set atom {atomID} x {rf2[0]} y {rf2[1]} z {rf2[2]}
                
                run 0
                write_data {xyz}
    ''')
    
    NEB_min(L1)
    
    rf2 = find_atom_position(L1,atomID)
    

    plottitle=f"Potential energy landscape around atom {atomID}"
    redXPts=np.transpose([[0,0]])
    allPts=[['x',redXPts]]
    
    if skipPES != 1 and me==0:
        plot_PES(PESimage,allPts,xlist,zlist,elist,plottitle)
    
    
    ####Now clean up the dump file to be the correct format for NEB runs
    if me == 0:## ONLY RUN ON ONE PROCESS
        
        with open(nebF,'w+') as f:
            f.write(f"1 \n{atomID} {rf2[0]} {rf2[1]} {rf2[2]}")
    return
    
if __name__ == "__main__":
    
    outfolder=sys.argv[1] 
    atomID=sys.argv[2]
    atom=sys.argv[6]
    filepath=sys.argv[7]
    plot= True if sys.argv[8] == 1 else False
    dumpstep=0
    nebFiles =prep_neb_into_iface(filepath,dumpstep,atomID,outfolder,fpos,plot)

    MPI.Finalize()
    exit()