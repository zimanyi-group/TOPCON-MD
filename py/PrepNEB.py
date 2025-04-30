#!/usr/bin/env python
"""
Author: Adam Goga
This script is a core piece of the NEB pipeline which is passed information about a single NEB run and sets up the initial and final data
files that are needed to further pass into the lammps NEB.lmp script. Cruicial to a NEB MD run is the fact that the initial and final data
structure are local energy minima thus this files main goal is to create those files by initiating a python LAMMPS script and minimizing the
correct locations.
"""
from lammps import lammps
import os

import numpy as np
from mpi4py import MPI

from random import gauss
from argparse import ArgumentParser



def find_atom_position(L,atomI):
    """
    Find an atoms position using the lammps python wrapper
    :param L - The lammps object.
    :param AtomI - The atom to find the position of.
    :return The position of the atom
    """
    cstr=f'''
        variable x{atomI} equal x[{atomI}]
        variable y{atomI} equal y[{atomI}]
        variable z{atomI} equal z[{atomI}]
        '''

    L.commands_string(cstr)
    
    x = L.extract_variable(f'x{atomI}')
    y = L.extract_variable(f'y{atomI}')
    z = L.extract_variable(f'z{atomI}')
    
    return (x,y,z)

def NEB_min(L):
    L.commands_string(f'''minimize {etol} {etol} 10000 10000''')


def get_id_list(L):
    """
    Extract the list of atom IDs from the simulation data.
    @param L - The lammps object.
    @return The list of atom IDs
    """
    
    nlocal = L.extract_global("nlocal")
    id_res=L.extract_atom('id')
    id_list= np.array([id_res[i] for i in range(nlocal)])
    return id_list


def init_dump(L,file,dumpstep):
    """
    Initialize a dump file using the lammps python library
    :param L - The lammps object.
    :param file - The dumpfile to load.
    :param dumpstep - the step to read from the dump
    :return None
    """
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
    """
    Initialize a data file using the lammps python library
    :param L - The lammps object.
    :param file - The datafile to load.
    :return None
    """
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
        run 0
        ''')

def create_ovito_plot(infile,figureName,atoms,infofile):
    """
    old function from before you were born
    """
    yslabwidth=7#ang
    try:

        pipeline = import_file(infile)
        y=0
        l=len(atoms)
        poslist=[]
        expr=""
        for i in range(l):
            #print(atoms[i])
            id=atoms[i][0]
            pos=np.array(atoms[i][1])

            poslist.append(pos)
            
            
            expr+=f'ParticleIdentifier=={id}'
            if i != l-1:
                expr+='||'

        pipeline.modifiers.append(ExpressionSelectionModifier(expression = expr))
        pipeline.modifiers.append(AssignColorModifier(color=(0, 1, 0)))

        fpos=poslist[-1]
        ipos=poslist[0]

        midpt=[(fpos[0]+ipos[0])/2,(fpos[1]+ipos[1])/2,(fpos[2]+ipos[2])/2]
        width=[(fpos[0]-ipos[0]),(fpos[1]-ipos[1]),(fpos[2]-ipos[2])]
  
        minwidths=[20,6,20]
        
        for i in range(len(width)):
            if width[i] < minwidths[i]:
                width[i]=minwidths[i]
            normal=[0,0,0]
            normal[i]=1
            pipeline.modifiers.append(ovito.modifiers.SliceModifier(normal=normal,distance=midpt[i],slab_width=width[i]))  
            

        vp = Viewport(type=Viewport.Type.Front)
        
        imagesize=(600,600)
        vp.zoom_all(size=imagesize)


        data=pipeline.compute()
        data.cell.vis.enabled = False  
        
        pipeline.add_to_scene()

        vp.zoom_all(size=imagesize)
        
        vp.render_image(size=imagesize, filename=figureName,renderer=TachyonRenderer(ambient_occlusion=False, shadows=False))
        
        pipeline.remove_from_scene()
        if me==0:
            #write info file
            with open(infofile,'a') as f:
                f.write(f"image {figureName}\n")
    except Exception as e:
        print(e)
        
def reduce_sim_box(L,rpos):
    """
    Cut off a lot of the simulation so that further runs can be faster. Generally we don't need all the atoms that are too far away
    to affect the atom we're interested in
    :param L - The lammps object.
    :param rpos - The posi
    :return None
    """
    xzhalfwidth = 5.1
    yhwidth=5.1
    step = .5
    buff=1

    xlist=np.arange(-xzhalfwidth,xzhalfwidth,step)
    zlist=np.arange(-xzhalfwidth,xzhalfwidth,step)

    xlen=len(xlist)
    zlen=len(zlist)
    
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
    """
    Create a potential energy surface
    :param L - The lammps object.
    :param atom - The atom to center the pes around.
    :return The file names of the initial position and the final neb xyz file
    """
    xzhalfwidth = 5.1
    yhwidth=5.1
    step = .5
    buff=1

    xlist=np.arange(-xzhalfwidth,xzhalfwidth,step)
    zlist=np.arange(-xzhalfwidth,xzhalfwidth,step)

    xlen=len(xlist)
    zlen=len(zlist)
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

    # class MidpointNormalize(mpl.colors.Normalize):
    #     def __init__(self, vmin, vmax, midpoint=0, clip=False):
    #         self.midpoint = midpoint
    #         mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    #     def __call__(self, value, clip=None):
    #         normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
    #         normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
    #         normalized_mid = 0.5
    #         x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
    #         return np.ma.masked_array(np.interp(value, x, y))

    # def plot_PES(PESimage,markerPts,xlist,zlist,elist,title):
    #     #Plotting below
    #     fig,ax = plt.subplots(figsize=(6,6))
        
    #      # set maximum value of the PES to be twice the lowest E or 18eV, whichever is highest
    #     minE=np.min(elist)
    #     maxE=np.max(elist)
    #     ab=maxE
        
        
    #     maxE=56
    #     minE=-8
        
    #     # if minE < -max:
    #     #     minE=-8
    #     # if maxE > max:
    #     #     maxE=max
            
    #     elist[elist<minE]=minE
    #     elist[elist>maxE]=maxE
    #     # minE=-maxE
    #     # maxE=2*abs(minE)
    #     # if maxE<max:
    #     #     maxE=max
    #     # elist[elist>maxE]=maxE
        
        
    #     # if finalLoc is not None:
    #     #     redXPts[1]=[finalLoc[0],finalLoc[1]]
    #     norm = MidpointNormalize(vmin=minE,vmax=maxE,midpoint=0)
    #     if maxE==0 and minE ==0:
    #         norm=None
            
    #     im=plt.contourf(zlist,xlist,elist,20,cmap='bwr',norm=norm)
        
    #     for m in markerPts:
    #         pts=m[1]
    #         plt.scatter(pts[0],pts[1],marker=m[0],c='g',)
            
    #     plt.grid('on',linewidth=0.25,linestyle='--')

    #     plt.axis('scaled')
    #     plt.xlabel('Δx(Å)')
    #     plt.ylabel('Δz(Å)',labelpad=0.05)
        
        
    #     divider = make_axes_locatable(ax)
    #     cax = divider.append_axes('right', size='5%', pad=0.05)
    #     cbar=fig.colorbar(im,cax=cax,orientation='vertical')
    #     cbar.set_label('ΔE(eV)')
    #     ax.set_title(title)
    #     #ax.set_xticklabels(np.arange(-math.floor(xzhalfwidth),math.floor(xzhalfwidth)+1,2))
    #     plt.savefig(PESimage)
        
    #     ###OLD working double PES ovito
    #     # fig = plt.figure(figsize=(12,6))
    #     # gs = gridspec.GridSpec(1, 2,width_ratios=[1,1.6])
    #     # ax1 = plt.subplot(gs[0])
    #     # ax2 = plt.subplot(gs[1])
        
    #     # #Load the atomistic view and put it in the second subplot
    #     # ovitoImage=plt.imread(ovitoFig)
    #     # ax2.axis('off')
    #     # ax2.imshow(ovitoImage,cmap='gray')
        
        
    #     # redXPts=np.transpose([[0,0],[rMin[0],rMin[1]]])
        
        
    #     # # if finalLoc is not None:
    #     # #     redXPts[1]=[finalLoc[0],finalLoc[1]]
            
    #     # im=ax1.contourf(zlist,xlist,elist,20,cmap='viridis')
    #     # ax1.scatter(redXPts[0],redXPts[1],marker='x',c='r')

    #     # ax1.axis('scaled')
    #     # ax1.set_xlabel('Δx(Å)')
    #     # ax1.set_ylabel('Δz(Å)')
        
        
    #     # divider = make_axes_locatable(ax1)
    #     # cax = divider.append_axes('right', size='5%', pad=0.15)
    #     # cbar=fig.colorbar(im,cax=cax,orientation='vertical')
    #     # cbar.set_label('ΔE(kcal/mol)')
    #     # ax1.set_title(f"Potential energy landscape around atom {atom}")
    #     # plt.savefig(PESimage)

def extract_box(L):
    """
    Pull the box out of a lammps python wrapper object
    """
    bbox=L.extract_box()
    return np.array([[bbox[0][0],bbox[1][0]],[bbox[0][1],bbox[1][1]],[bbox[0][2],bbox[1][2]]])


def recenter_sim(L,r):
    """
    Center the simulation on a point/atom
    :param L - The lammps object.
    :param r - The position to center around
    :return The new box
    """
    bbox= extract_box(L)

    xhlen=abs(bbox[0][1]-bbox[0][0])/2
    yhlen=abs(bbox[1][1]-bbox[1][0])/2
    zhlen=abs(bbox[2][1]-bbox[2][0])/2
    # print(xhlen)
    # print(xhlen-r[0])
    
    L.commands_string(f'''
        
        displace_atoms all move {xhlen-r[0]} {yhlen-r[1]} {zhlen-r[2]}
        run 0''')
    
    return bbox



def get_lammps(log):
    """
    Create a new lammps python wrapper simulation
    """
    return lammps('mpi',["-log",log,'-screen','none'])


def place_random_O(L,zlims,seed):
    """
    Test function please ignore
    """
    
    L.commands_string(f''' 
                    region r_randO block EDGE EDGE EDGE EDGE {zlims[0]} {zlims[1]}
                    create_atoms 2 random 1 {seed} r_randO overlap 1.0 maxtry 1000
                    # group new_atom empty
                    # fix fdep all deposit 1 2 1 {seed} region r_randO id max near 2
                    # run 1
                      ''')
    

def prep_neb_zap_multi(file,dumpstep,atomI,aditionalAtoms,atomF,outfolder,infofile,plot,skipPES=True):
    """
    Prepare the NEB (Nudged Elastic Band) calculation for moving multiple atoms to the location of one atom that has been zapped.
    Create a log file in the proper folder which passed details of this NEB to the rest of the pipeline.
    @param file - the input file
    @param dumpstep - the dumpstep to load
    @param atomI - The main atom that will be moved
    @param aditionalAtoms - additional atoms to be moved with the main atom
    @param atomF - final atom which gets zapped
    @param outfolder - output folder
    @param infofile - information file that passed the details of the run to the NEB.lmp file and Process-NEB.py 
    @param plot - whether to plot the results
    @param skipPES - flag to skip PES calculation
    @return None
    """
    
    print(f"{atomI}-{aditionalAtoms[0]} zapping {atomF}")
    # plt.rcParams["figure.autolayout"] = True
    
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
    
    extraMovers=[]
    for aa in aditionalAtoms:
        rai = find_atom_position(L1,aa)
        #get the seperation vector between the mover and any extra movers
        tpos = [rai[0]-ri[0],rai[1]-ri[1],rai[2]-ri[2]]
        extraMovers.append([aa,tpos])
        
    if me==0:
        #write info file
        with open(infofile,'a') as f:
            f.write(f"multizap {len(aditionalAtoms)+1}\n")
            f.write(f"neb 0 {atomI} {fileIdent} {nebI} {nebF} {fileIdent}-{atomF}.log {extraMovers[0][0]}\n"+#bash file is expecting a h atom after log file
                    f"initial-dat-0 {nebI}\n"+
                    f"final-dat-0 {nebF}\n"
                    )
    
    
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
    
    if me==0:
        #now write to the info file
        with open(infofile,'a') as f:
            f.write(f"pcsv_{fileIdent}_iPos [{icoord[0]},{icoord[1]},{icoord[2]}]\n")
            f.write(f"pcsv_{fileIdent}_fPos [{fcoord[0]},{fcoord[1]},{fcoord[2]}]\n")
            f.write(f"pcsv_{fileIdent}_box [[{bbox[0][0]},{bbox[0][1]}],[{bbox[1][0]},{bbox[1][1]}],[{bbox[2][0]},{bbox[2][1]}]]\n")
    
    
    
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
                set atom {atomI} x {rf2[0]} y {rf2[1]} z {rf2[2]}
    ''')
    
    for ex in extraMovers:
        L2f.commands_string(f'''
                set atom {ex[0]} x {rf2[0]+ex[1][0]} y {rf2[1]+ex[1][1]} z {rf2[2]+ex[1][2]}
        ''')
    
    L2f.commands_string(f'''       
                run 0
                write_data {xyz}
    ''')
    
    NEB_min(L2f)
    
    rf2 = find_atom_position(L2f,atomI)
    
    finalPosAdd=[]
    for aa in aditionalAtoms:

        raf= find_atom_position(L2f,aa)
        print(raf)
        finalPosAdd.append([aa,raf])
        

    plottitle=f"Potential energy landscape around atom {atomI}"
    redXPts=np.transpose([[0,0]])
    allPts=[['x',redXPts]]
    
    # if skipPES != 1 and me==0:
    #     plot_PES(PESimage,allPts,xlist,zlist,elist,plottitle)
    
    
    ####Now clean up the dump file to be the correct format for NEB runs
    if me == 0:## ONLY RUN ON ONE PROCESS
        
        with open(nebF,'w+') as f:
            numAdd=len(aditionalAtoms)
            f.write(f'''{numAdd+1}\n{atomI} {rf2[0]} {rf2[1]} {rf2[2]}''')
            for a in finalPosAdd:
                f.write(f"\n{a[0]} {a[1][0]} {a[1][1]} {a[1][2]}")
            
            
                
    return

def prep_neb_multi_jump(args):
    """
    Prepare the NEB (Nudged Elastic Band) calculation for a multi-jump, i.e. take one atom and jump it to a number of different locations
    Create a log file in the proper folder which passed details of this NEB to the rest of the pipeline.
    @param args - needs to have the following information 
        @param dfile - the input file
        @param atomI - The main atom that will be moved
        @param out - output folder
        @param info - information file that passed the details of the run to the NEB.lmp file and Process-NEB.py 
        @param bclist - The list of bond centers which the atom will be moved to.
    @return None
    """
    currentIfile=nebI=args.dfile
    atomI=args.atomid
    outfolder=args.out
    infofile=args.info

    bondcenter_list=args.bclist.split(',')#list of pairs of atoms whose bond center is the final position
    i=9
    
    # for j in range(numjumps):
    #     bondcenter_list.append([int(sys.argv[i+j]),int(sys.argv[i+j+1])])
    #     i+=1
        
    jumps=len(bondcenter_list)
    
    with open(infofile,'a') as f:
        if me==0:
            print(f"Num jumps {jumps}")
            f.write(f"multijump {jumps}\n")

        
        for i in range(jumps):
            identifier=f"j{i}"
            bc=bondcenter_list[i].split()
            ba1=int(bc[0])
            ba2=int(bc[1])
            

            # # fposx=bc[0]
            # # fposy=bc[1]
            # # fposz=bc[2]
            
            # fpos=[fposx,fposy,fposz]
            
            (nebI, nebF,nebFfullData) =prep_neb_to_bond_center(args,identifier,ba1,ba2,datafile=currentIfile,write_info=False)
            
            if me==0:
                f.write(f"neb {i} {atomI} {identifier} {nebI} {nebF} {identifier}-{atomI}.log {atomI}\n"+#bash file is expecting a h atom after log file
                        f"{identifier}-atoms {bc[0]} {bc[1]}\n"+
                        f"initial-dat-{i} {nebI}\n"+
                        f"final-dat-{i} {nebF}\n"
                        )
            currentIfile=nebFfullData#start the next jump where the last left off
        
def midpt(p1,p2):
    """
    Simple midpoint calculation
    """
    midpt=[0,0,0]
    for i in range(len(p1)):
        midpt[i]=(p1[i]+p2[i])/2
    return midpt


def prep_neb_to_bond_center(args,ident,atomF1=None,atomF2=None,datafile=None,dumpstep=0,write_info=True):
    """
    Prepare the NEB (Nudged Elastic Band) calculation to a bond center(BC) given by two input atoms.
    Create a log file in the proper folder which passed details of this NEB to the rest of the pipeline.
    @param args - the arguments for the calculation must contain
        @param atomid - The atom doing the moving
        @param out - the output folder
    @param ident - the identifier string for the calculation
    @param atomF1 - the first atom of the final BC
    @param atomF2 - the second atom of the final BC
    @param datafile - the data file for the calculation
    @param dumpstep - the dump step for the input file
    @param write_info - flag whether this function handles writing the info or another
    @return None
    """
    outfolder=args.out
    atomI=args.atomid
    
    if atomF1 is None or atomF2 is None:
        atomF1=args.bc1
        atomF2=args.bc2
        
    infofile=args.info
    if datafile is None:
        datafile=args.dfile
        
    L1 = get_lammps(f'{outfolder}/logs/PrepNEB-If.log')
    
    fileIdent=f'{atomI}-{ident}'
    
    full= outfolder+ f'{fileIdent}-Full.data'
    
    #initilize the data files 
    if datafile.endswith(".dump"):
        LT = get_lammps(f'{outfolder}/logs/PrepNEB-LT.log')
        #do this first initialize to get around reaxff issues(charge stuff I think)
        init_dump(LT,datafile,dumpstep)
        LT.commands_string(f'''
            write_data {full}
            ''')
        #
        init_dat(L1,full)
        
    elif datafile.endswith(".data") or datafile.endswith(".dat"):
        init_dat(L1,datafile)
    else:
        print("File is not a .data or .dump")
        return

    rf1 = find_atom_position(L1,atomF1)
    rf2 = find_atom_position(L1,atomF2)
    
    midpoint=midpt(rf1,rf2)
    if me==0:
        print(f"prep_neb_to_bond_center - {rf1} {rf2} {midpoint}")
    return prep_neb_to_location(args,ident,fileIdent,midpoint,L1=L1,dumpstep=dumpstep,write_info=write_info)
    

def prep_neb_h_to_h2_region(args,fileIdent,ident,fpos,datafile=None,L1=None,dumpstep=0,write_info=True):
    """
    Given an H atom we will create a region a certain distance away from the interface, lammps will be used to create an atom at a random location
    inside that region and we will then move the H atom to a location nearby that atom. Create a log file in the proper folder which passed details
    of this NEB to the rest of the pipeline.
    @param args - the arguments for the calculation must contain
        @param atomid - The atom doing the moving
        @param out - the output folder
    @param ident - the identifier string for the calculation
    @param fpos - the final position of the h2 atom
    @param L1 - a lammps python wrapper that has already loaded the datafile
    @param datafile - the data file for the calculation
    @param dumpstep - the dump step for the input file
    @param write_info - flag whether this function handles writing the info or another
    @return None
    """

    outfolder=args.out
    atomI=args.atomid
    if datafile is None:
        datafile=args.dfile

    infofile=args.info
    # plt.rcParams["figure.autolayout"] = True
    nebI=outfolder+f'{fileIdent}-NEBI.data'
    nebF=outfolder+f'{fileIdent}-NEBF.data'
    
    if me==0 and write_info:
        #now write to the info file
        with open(infofile,'a') as f:
            f.write(f"neb 0 {atomI} {ident} {nebI} {nebF} {ident}-{atomI}.log {atomI}\n")
    
    if L1 is None:
        L1 = get_lammps(f'{outfolder}/logs/PrepNEB-L1.log')
    else:
        L1.commands_string(f'''
            clear
            ''')
        
    full= outfolder+ f'{fileIdent}-Full.data'
    if datafile.endswith(".dump"):
        LT = get_lammps(f'{outfolder}/logs/PrepNEB-LT.log')
        #do this first initialize to get around reaxff issues(charge stuff I think)
        init_dump(LT,datafile,dumpstep)
        LT.commands_string(f'''
            write_data {full}
            ''')
        init_dat(L1,full)
        
    elif datafile.endswith(".data") or datafile.endswith(".dat"):
        init_dat(L1,datafile)
    else:
        print("File is not a .data or .dump")
        return

    
    
    
    # PESimage=outfolder+f"PES({fileIdent}).png"
    
    bbox=extract_box(L1)
    ##### L1 - create the initial NEB data file
    ri = find_atom_position(L1,atomI)

    icoord=ri#have to use the coordinates before recentering
    
    r_half_width=.5

    b_id_list=get_id_list(L1)


    r_xpos=float(fpos[0])
    r_ypos=float(fpos[1])
    r_zpos=float(fpos[2])
    L1.commands_string(f'''
        region h2 block {r_xpos-r_half_width} {r_xpos+r_half_width} {r_ypos-r_half_width} {r_ypos+r_half_width} {r_zpos-r_half_width} {r_zpos+r_half_width}
        create_atoms 3 random 1 12345 h2 overlap 1 maxtry 10
    ''')
    
    a_id_list=get_id_list(L1)

    new_h_id=None
    for ai in a_id_list:
        if ai not in b_id_list:
            new_h_id=ai
            
            
            
            
    NEB_min(L1)
    
    
    
    print(f"New h atom created has id {new_h_id}")

    

    bbox=recenter_sim(L1,ri)

    
    if new_h_id is not None:
        ri = np.array(find_atom_position(L1,atomI))
        new_h_pos = np.array(find_atom_position(L1,new_h_id))
        
        sep_vec = new_h_pos - ri
        l=0
        for i in range(3):
            l+=sep_vec[i]**2
        len_sep=l**(1/2)
        print(f"{ri} -> {new_h_pos} - {sep_vec} of len {len_sep}")
        sep_vec/=len_sep
    else:
        print(f"Big problems in prep_neb_h_to_h2_region")
        return 
    
    h2_blen=.78#angstroms
    fpos = ri + (len_sep-h2_blen)*sep_vec
    
    
    L1.commands_string(f'''
        write_data {nebI}
    ''')
    
    ri = find_atom_position(L1,atomI)
     
    # if me == 0 and plot:
    #     #Now create ovito plot of atoms for future use
    #     create_ovito_plot(nebI,ovitoFig,ri,atomI,selection)
    
    
    if me==0:
        #now write to the info file
        with open(infofile,'a') as f:
            f.write(f"initial_coords_{ident} {icoord[0]} {icoord[1]} {icoord[2]}\n")
            f.write(f"pcsv_{ident}_iPos [{icoord[0]},{icoord[1]},{icoord[2]}]\n")
            f.write(f"pcsv_{ident}_fPos [{fpos[0]},{fpos[1]},{fpos[2]}]\n")
            f.write(f"pcsv_{ident}_box [[{bbox[0][0]},{bbox[0][1]}],[{bbox[1][0]},{bbox[1][1]}],[{bbox[2][0]},{bbox[2][1]}]]\n")

    
    xyz=outfolder+f'{fileIdent}-NEBFXYZ.data'

        
    #now create the lowest energy position data file for NEB.
    L1.commands_string(f'''
                set atom {atomI} x {fpos[0]} y {fpos[1]} z {fpos[2]}
    ''')
    NEB_min(L1)


    L1.commands_string(f'''
                run 0
                write_data {xyz}
                ''')
    
    rf = find_atom_position(L1,atomI)
    
    L1.commands_string(f'''
    region r_local sphere {fpos[0]} {fpos[1]} {fpos[2]} 2.5
    group g_local region r_local

    ''') 
    
    L1.commands_string(f'''
    dump d1 g_local custom 1 {outfolder}/test.dump id x y z
    run 0
    '''
    )
    
    # ids = np.array(L1.extract_variable("local_i","g_local"))
    # xs = np.array(L1.extract_variable("local_x","g_local"))
    # ys = np.array(L1.extract_variable("local_y","g_local"))
    # zs = np.array(L1.extract_variable("local_z","g_local"))
    
    
    # id_list=ids[ids!=0]
    # x_list=xs[xs!=0]
    # y_list=ys[ys!=0]
    # z_list=zs[zs!=0]

    # if me==0:
    #     for i in ids:
    #         print(i)
    
    # print(f"Initial POS for atom{atomI}: {ri}")
    # print(f"Final POS for atom{atomI}: {rf}")
    
    
    ####Now clean up the dump file to be the correct format for NEB runs
    if me == 0:## ONLY RUN ON ONE PROCESS
        # with open(nebF,'w+') as f:
        #     f.write(f"1 \n{atomI} {rf[0]} {rf[1]} {rf[2]}")
        outstr=""
        with open(f"{outfolder}/test.dump") as f:
            next = False
            rest = False
            for line in f:
 
                
                if next:
                    outstr+=line
                    next = False
                if rest:
                    outstr+=line
                
                if line == "ITEM: NUMBER OF ATOMS\n":
                    next = True
                if line == "ITEM: ATOMS id x y z\n":
                    rest = True

        with open(nebF,'w+') as f:
            f.write(outstr)
            # f.write(f"{len(id_list)} \n") 
            # for i in range(len(id_list)):
            #     id=int(id_list[i])
            #     x=x_list[i]
            #     y=y_list[i]
            #     z=z_list[i]
            #     f.write(f"{id} {x} {y} {z} \n")
    return (nebI,nebF,xyz)


def prep_neb_h_to_h2_location(args,fileIdent,ident,fpos,h2pos,datafile=None,L1=None,dumpstep=0,write_info=True):
    """
    Given an H atom and two positions we make an additional 'interstitial' h atom at the firstlocation and then move the original h
        atom to the second location which should be forming a H2 pair. Create a log file in the proper folder which passed details of 
        this NEB to the rest of the pipeline.
    @param args - the arguments for the calculation must contain
        @param atomid - The atom doing the moving
        @param out - the output folder
    @param ident - the identifier string for the calculation
    @param fpos - the final position of the initial h atom
    @param h2pos - final position of second h atom we will create
    @param datafile - the data file for the calculation
    @param dumpstep - the dump step for the input file
    @param write_info - flag whether this function handles writing the info or another
    @return None
    """
    outfolder=args.out
    atomI=args.atomid
    if datafile is None:
        datafile=args.dfile

    infofile=args.info
    # plt.rcParams["figure.autolayout"] = True
    nebI=outfolder+f'{fileIdent}-NEBI.data'
    nebF=outfolder+f'{fileIdent}-NEBF.data'
    
    if me==0 and write_info:
        #now write to the info file
        with open(infofile,'a') as f:
            f.write(f"neb 0 {atomI} {ident} {nebI} {nebF} {ident}-{atomI}.log {atomI}\n")
    
    if L1 is None:
        L1 = get_lammps(f'{outfolder}/logs/PrepNEB-L1.log')
    else:
        L1.commands_string(f'''
            clear
            ''')
        
    full= outfolder+ f'{fileIdent}-Full.data'
    if datafile.endswith(".dump"):
        LT = get_lammps(f'{outfolder}/logs/PrepNEB-LT.log')
        #do this first initialize to get around reaxff issues(charge stuff I think)
        init_dump(LT,datafile,dumpstep)
        LT.commands_string(f'''
            write_data {full}
            ''')
        init_dat(L1,full)
        
    elif datafile.endswith(".data") or datafile.endswith(".dat"):
        init_dat(L1,datafile)
    else:
        print("File is not a .data or .dump")
        return

    
    
    
    # PESimage=outfolder+f"PES({fileIdent}).png"
    
    bbox=extract_box(L1)
    ##### L1 - create the initial NEB data file
    ri = find_atom_position(L1,atomI)

    icoord=ri#have to use the coordinates before recentering
    
    
    L1.commands_string(f'''
        create_atoms 3 single {h2pos[0]} {h2pos[1]} {h2pos[2]}
    ''')
    
    # bbox=recenter_sim(L1,ri)
    #ri = find_atom_position(L1,atomI)
    NEB_min(L1)
    
    
    
    L1.commands_string(f'''
        write_data {nebI}
    ''')
    
    ri = find_atom_position(L1,atomI)
     
    # if me == 0 and plot:
    #     #Now create ovito plot of atoms for future use
    #     create_ovito_plot(nebI,ovitoFig,ri,atomI,selection)
    
    
    if me==0:
        #now write to the info file
        with open(infofile,'a') as f:
            f.write(f"initial_coords_{ident} {icoord[0]} {icoord[1]} {icoord[2]}\n")
            f.write(f"pcsv_{ident}_iPos [{icoord[0]},{icoord[1]},{icoord[2]}]\n")
            f.write(f"pcsv_{ident}_fPos [{fpos[0]},{fpos[1]},{fpos[2]}]\n")
            f.write(f"pcsv_{ident}_box [[{bbox[0][0]},{bbox[0][1]}],[{bbox[1][0]},{bbox[1][1]}],[{bbox[2][0]},{bbox[2][1]}]]\n")

    
    xyz=outfolder+f'{fileIdent}-NEBFXYZ.data'

        
    #now create the lowest energy position data file for NEB.
    L1.commands_string(f'''
                set atom {atomI} x {fpos[0]} y {fpos[1]} z {fpos[2]}
    ''')
    NEB_min(L1)


    L1.commands_string(f'''
                run 0
                write_data {xyz}
                ''')
    
    rf = find_atom_position(L1,atomI)
    
    L1.commands_string(f'''
    region r_local sphere {fpos[0]} {fpos[1]} {fpos[2]} 2.5
    group g_local region r_local

    variable local_i atom id
    variable local_x atom x
    variable local_y atom y
    variable local_z atom z
    ''') 
    
    L1.commands_string(f'''
    dump d1 g_local custom 1 {outfolder}/test.dump id x y z
    run 0
    '''
    )
    
    ids = np.array(L1.extract_variable("local_i","g_local"))
    xs = np.array(L1.extract_variable("local_x","g_local"))
    ys = np.array(L1.extract_variable("local_y","g_local"))
    zs = np.array(L1.extract_variable("local_z","g_local"))
    
    
    id_list=ids[ids!=0]
    x_list=xs[xs!=0]
    y_list=ys[ys!=0]
    z_list=zs[zs!=0]

    # if me==0:
    #     for i in ids:
    #         print(i)
    
    # print(f"Initial POS for atom{atomI}: {ri}")
    # print(f"Final POS for atom{atomI}: {rf}")
    
    
    ####Now clean up the dump file to be the correct format for NEB runs
    if me == 0:## ONLY RUN ON ONE PROCESS
        # with open(nebF,'w+') as f:
        #     f.write(f"1 \n{atomI} {rf[0]} {rf[1]} {rf[2]}")
        outstr=""
        with open(f"{outfolder}/test.dump") as f:
            next = False
            rest = False
            for line in f:
 
                
                if next:
                    outstr+=line
                    next = False
                if rest:
                    outstr+=line
                
                if line == "ITEM: NUMBER OF ATOMS\n":
                    next = True
                if line == "ITEM: ATOMS id x y z\n":
                    rest = True

        with open(nebF,'w+') as f:
            f.write(outstr)
            # f.write(f"{len(id_list)} \n") 
            # for i in range(len(id_list)):
            #     id=int(id_list[i])
            #     x=x_list[i]
            #     y=y_list[i]
            #     z=z_list[i]
            #     f.write(f"{id} {x} {y} {z} \n")
    return (nebI,nebF,xyz)

def prep_neb_to_location(args,fileIdent,ident,fpos,datafile=None,L1=None,dumpstep=0,write_info=True):
    """
    The main workhorse function, takes an atom and moves it to a location taking care to minimize the system and create
        proper data files for the initial and final state
        
    @param args - the arguments for the calculation must contain
        @param atomid - The atom doing the moving
        @param out - the output folder
    @param ident - the identifier string for the calculation
    @param fpos - the final position of the moving atom
    @param datafile - the data file for the calculation
    @param L1 - A possible instance of the lammps file that may or may not need to be started
    @param dumpstep - the dump step for the input file
    @param write_info - flag whether this function handles writing the info or another
    @return None
    """
    outfolder=args.out
    atomI=args.atomid
    if datafile is None:
        datafile=args.dfile

    infofile=args.info
    # plt.rcParams["figure.autolayout"] = True
    nebI=outfolder+f'{fileIdent}-NEBI.data'
    nebF=outfolder+f'{fileIdent}-NEBF.data'
    
    if me==0 and write_info:
        #now write to the info file
        with open(infofile,'a') as f:
            f.write(f"neb 0 {atomI} {ident} {nebI} {nebF} {ident}-{atomI}.log {atomI}\n")
    
    if L1 is None:
        L1 = get_lammps(f'{outfolder}/logs/PrepNEB-L1.log')
    else:
        L1.commands_string(f'''
            clear
            ''')
        
    full= outfolder+ f'{fileIdent}-Full.data'
    if datafile.endswith(".dump"):
        LT = get_lammps(f'{outfolder}/logs/PrepNEB-LT.log')
        #do this first initialize to get around reaxff issues(charge stuff I think)
        init_dump(LT,datafile,dumpstep)
        LT.commands_string(f'''
            write_data {full}
            ''')
        init_dat(L1,full)
        
    elif datafile.endswith(".data") or datafile.endswith(".dat"):
        init_dat(L1,datafile)
    else:
        print("File is not a .data or .dump")
        return

    
    
    
    # PESimage=outfolder+f"PES({fileIdent}).png"
    
    bbox=extract_box(L1)
    ##### L1 - create the initial NEB data file
    ri = find_atom_position(L1,atomI)

    icoord=ri#have to use the coordinates before recentering
    
    # bbox=recenter_sim(L1,ri)
    #ri = find_atom_position(L1,atomI)
    NEB_min(L1)
    L1.commands_string(f'''
        write_data {nebI}
    ''')
    
    ri = find_atom_position(L1,atomI)
     
    # if me == 0 and plot:
    #     #Now create ovito plot of atoms for future use
    #     create_ovito_plot(nebI,ovitoFig,ri,atomI,selection)
    
    
    if me==0:
        #now write to the info file
        with open(infofile,'a') as f:
            f.write(f"initial_coords_{ident} {icoord[0]} {icoord[1]} {icoord[2]}\n")
            f.write(f"pcsv_{ident}_iPos [{icoord[0]},{icoord[1]},{icoord[2]}]\n")
            f.write(f"pcsv_{ident}_fPos [{fpos[0]},{fpos[1]},{fpos[2]}]\n")
            f.write(f"pcsv_{ident}_box [[{bbox[0][0]},{bbox[0][1]}],[{bbox[1][0]},{bbox[1][1]}],[{bbox[2][0]},{bbox[2][1]}]]\n")

    
    xyz=outfolder+f'{fileIdent}-NEBFXYZ.data'

        
    #now create the lowest energy position data file for NEB.
    L1.commands_string(f'''
                set atom {atomI} x {fpos[0]} y {fpos[1]} z {fpos[2]}
    ''')
    NEB_min(L1)


    L1.commands_string(f'''
                run 0
                write_data {xyz}
                ''')
    
    rf = find_atom_position(L1,atomI)
    
    L1.commands_string(f'''
    region r_local sphere {fpos[0]} {fpos[1]} {fpos[2]} 2.5
    group g_local region r_local

    variable local_i atom id
    variable local_x atom x
    variable local_y atom y
    variable local_z atom z
    ''') 
    
    L1.commands_string(f'''
    dump d1 g_local custom 1 {outfolder}/test.dump id x y z
    run 0
    '''
    )
    
    ids = np.array(L1.extract_variable("local_i","g_local"))
    xs = np.array(L1.extract_variable("local_x","g_local"))
    ys = np.array(L1.extract_variable("local_y","g_local"))
    zs = np.array(L1.extract_variable("local_z","g_local"))
    
    
    id_list=ids[ids!=0]
    x_list=xs[xs!=0]
    y_list=ys[ys!=0]
    z_list=zs[zs!=0]

    # if me==0:
    #     for i in ids:
    #         print(i)
    
    # print(f"Initial POS for atom{atomI}: {ri}")
    # print(f"Final POS for atom{atomI}: {rf}")
    
    
    ####Now clean up the dump file to be the correct format for NEB runs
    if me == 0:## ONLY RUN ON ONE PROCESS
        # with open(nebF,'w+') as f:
        #     f.write(f"1 \n{atomI} {rf[0]} {rf[1]} {rf[2]}")
        outstr=""
        with open(f"{outfolder}/test.dump") as f:
            next = False
            rest = False
            for line in f:
 
                
                if next:
                    outstr+=line
                    next = False
                if rest:
                    outstr+=line
                
                if line == "ITEM: NUMBER OF ATOMS\n":
                    next = True
                if line == "ITEM: ATOMS id x y z\n":
                    rest = True

        with open(nebF,'w+') as f:
            f.write(outstr)
            # f.write(f"{len(id_list)} \n") 
            # for i in range(len(id_list)):
            #     id=int(id_list[i])
            #     x=x_list[i]
            #     y=y_list[i]
            #     z=z_list[i]
            #     f.write(f"{id} {x} {y} {z} \n")
    return (nebI,nebF,xyz)
    

def prep_neb_to_location_boomerang(args,fileIdent,ident,fpos,datafile=None,L1=None,dumpstep=0,write_info=True,boomerang=10):
    """
    The main workhorse function, takes an atom and moves it to a location taking care to minimize the system and create
        proper data files for the initial and final state. The boomerang version of this moves the atom to the final pos
        and back to the initial pos a given number of times before finally creating the initial and final data files for NEB.
        This was a test run that I was doing to determine if this significantly changed the results of the NEB calculation
        
    @param args - the arguments for the calculation must contain
        @param atomid - The atom doing the moving
        @param out - the output folder
    @param ident - the identifier string for the calculation
    @param fpos - the final position of the moving atom
    @param datafile - the data file for the calculation
    @param L1 - A possible instance of the lammps file that may or may not need to be started
    @param dumpstep - the dump step for the input file
    @param write_info - flag whether this function handles writing the info or another
    @param boomerang - Number of times to boomerang
    @return None
    """
    outfolder=args.out
    atomI=args.atomid
    if datafile is None:
        datafile=args.dfile

    infofile=args.info
    # plt.rcParams["figure.autolayout"] = True
    nebI=outfolder+f'{fileIdent}-NEBI.data'
    nebF=outfolder+f'{fileIdent}-NEBF.data'
    
    
    if L1 is None:
        if me==0 and write_info:
            #now write to the info file
            with open(infofile,'a') as f:
                f.write(f"neb 0 {atomI} {ident} {nebI} {nebF} {ident}-{atomI}.log {atomI}\n")
        
        L1 = get_lammps(f'{outfolder}/logs/PrepNEB-L1.log')
        full= outfolder+ f'{fileIdent}-Full.data'
        if datafile.endswith(".dump"):
            LT = get_lammps(f'{outfolder}/logs/PrepNEB-LT.log')
            #do this first initialize to get around reaxff issues(charge stuff I think)
            init_dump(LT,datafile,dumpstep)
            LT.commands_string(f'''
                write_data {full}
                ''')
            init_dat(L1,full)
            
        elif datafile.endswith(".data") or datafile.endswith(".dat"):
            init_dat(L1,datafile)
        else:
            print("File is not a .data or .dump")
            return
    else:
        L1.commands_string(f'''
                clear
                ''')
    
    
    # PESimage=outfolder+f"PES({fileIdent}).png"
    
    bbox=extract_box(L1)
    ##### L1 - create the initial NEB data file
    ri = find_atom_position(L1,atomI)

    icoord=ri#have to use the coordinates before recentering

    xyz=outfolder+f'{fileIdent}-NEBFXYZ.data'
    

    for bi in range(boomerang-1):
        
        #now create the lowest energy position data file for NEB.
        L1.commands_string(f'''
                    set atom {atomI} x {fpos[0]} y {fpos[1]} z {fpos[2]}
        ''')
        NEB_min(L1)
        
        if me == 0 and os.path.exists(xyz):
            os.remove(xyz)

        #now create the lowest energy position data file for NEB.
        L1.commands_string(f'''
                    run 0
                    write_data {xyz} 
                    clear
        ''')
        
        init_dat(L1,xyz)
        NEB_min(L1)
        L1.commands_string(f'''
                    set atom {atomI} x {icoord[0]} y {icoord[1]} z {icoord[2]}
        ''')
        NEB_min(L1)

        if me == 0 and os.path.exists(xyz):
            os.remove(xyz)
            
            
        L1.commands_string(f'''
                    run 0
                    write_data {xyz}
                    clear
        ''')
        init_dat(L1,xyz)
        NEB_min(L1)
    
    
    
    #now create the lowest energy position data file for NEB.
    L1.commands_string(f'''
                set atom {atomI} x {fpos[0]} y {fpos[1]} z {fpos[2]}
    ''')
    NEB_min(L1)
    
    if me == 0 and os.path.exists(xyz):
        os.remove(xyz)

    #now create the lowest energy position data file for NEB.
    L1.commands_string(f'''
                run 0
                write_data {xyz} 
                clear
    ''')
    
    init_dat(L1,xyz)
    NEB_min(L1)
    L1.commands_string(f'''
                set atom {atomI} x {icoord[0]} y {icoord[1]} z {icoord[2]}
    ''')
    NEB_min(L1)

    if me == 0 and os.path.exists(xyz):
        os.remove(xyz)
        
        
    L1.commands_string(f'''
                run 0
                write_data {xyz}
                clear
    ''')
    init_dat(L1,xyz)
    NEB_min(L1)
        
        
    
    
    # bbox=recenter_sim(L1,ri)
    ri = find_atom_position(L1,atomI)
    NEB_min(L1)
    L1.commands_string(f'''
        write_data {nebI}
    ''')
    
    ri = find_atom_position(L1,atomI)
     

    
    L1.commands_string(f'''
                set atom {atomI} x {fpos[0]} y {fpos[1]} z {fpos[2]}
    ''')
    NEB_min(L1)
    
    rf = find_atom_position(L1,atomI)
    
    if me==0 and write_info:
        #now write to the info file
        with open(infofile,'a') as f:
            
            f.write(f"pcsv_{ident}_iPos [{icoord[0]},{icoord[1]},{icoord[2]}]\n")
            f.write(f"pcsv_{ident}_fPos [{fpos[0]},{fpos[1]},{fpos[2]}]\n")
            f.write(f"pcsv_{ident}_box [[{bbox[0][0]},{bbox[0][1]}],[{bbox[1][0]},{bbox[1][1]}],[{bbox[2][0]},{bbox[2][1]}]]\n")

    
    ####Now clean up the dump file to be the correct format for NEB runs
    if me == 0:## ONLY RUN ON ONE PROCESS
        
        with open(nebF,'w+') as f:
            f.write(f"1 \n{atomI} {rf[0]} {rf[1]} {rf[2]}")
    return (nebI,nebF,xyz)

def prep_neb_zap_single(args):
    """
    Prepare the NEB (Nudged Elastic Band) calculation for a single zap run i.e. take one atom and a nearby atom, delete the nearby atom,
    and use the final location of the deleted atom as the final position of the moving atom. This is primarily a worker func for multi_zap runs
    Create a log file in the proper folder which passed details of this NEB to the rest of the pipeline.
    @param args - needs to have the following information 
        @param dfile - the input file
        @param atomI - The main atom that will be moved
        @param atomF - The atom that will be deleted and moved into.
        @param out - output folder
        @param info - information file that passed the details of the run to the NEB.lmp file and Process-NEB.py 
        @param plot - whether to plot (deprecated)
    @return The initial and final position, plus the file of the initial location
    """
    outfolder=args.out
    atomI=args.atomid
    atomF=args.zapid
    infofile=args.info
    file=args.dfile
    plot=args.plot
    dumpstep=0
    skipPES=0
    
    
    # plt.rcParams["figure.autolayout"] = True
    
    #Need two lammps instances so that when removing an atom and minimizing we don't increase time for final NEB image minimization
    L1 = get_lammps(f'{outfolder}/logs/PrepNEB-I.log')
    L2 = get_lammps(f'{outfolder}/logs/PrepNEB-F.log')

    
    fileIdent=f'{atomI}-{atomF}'

    reset1=outfolder+f'{fileIdent}-NEBI.dump'
    reset2=outfolder+f'{fileIdent}-NEBF.dump'
    nebI=outfolder+f'{fileIdent}-NEBI.data'
    nebF=outfolder+f'{fileIdent}-NEBF.data'
    full= outfolder+ f'{fileIdent}-Full.data'
    
    PESimage="/home/adam/code/topcon-md/neb-out/"+f"PES({fileIdent}).png"
    ovitoFig=outfolder+f"{fileIdent}-Ovito.png"
    
    selection=[atomI,atomF]
    
    
    #initilize the data files 
    if file.endswith(".dump"):
        LT = get_lammps(f'{outfolder}/logs/PrepNEB-LT.log')
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

    if me==0:
        #write info file
        with open(infofile,'a') as f:
            f.write(f"zap\n")
            f.write(f"neb 0 {atomI} {fileIdent} {nebI} {nebF} {fileIdent}.log {atomI}\n"+#bash file is expecting a h atom after log file
                    f"initial-dat-0 {nebI}\n"+
                    f"final-dat-0 {nebF}\n"
                    )
    
    
    icoord=ri#have to use the coordinates before recentering
    fcoord=rf
    #print(L1.extract_compute('thermo_pe',0,0)*conv)
    bbox=recenter_sim(L1,ri)
    #print(L1.extract_compute('thermo_pe',0,0)*conv)

    #ri = find_atom_position(L1,atomI)
    NEB_min(L1)
    r_initial = find_atom_position(L1,atomI)
    
    L1.commands_string(f'''
    write_data {nebI}
    ''')
    
    ri = find_atom_position(L1,atomI)
    rf = find_atom_position(L1,atomF)
     
    if me == 0 and plot:
        #Now create ovito plot of atoms for future use
        atoms=[[atomI,ri],[atomF,rf]]
        #create_ovito_plot(nebI,ovitoFig,atoms,infofile)
    
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
    L1f = get_lammps(f'{outfolder}/logs/PrepNEB-If.log')
    init_dump(L1f,reset1,0)
    NEB_min(L1f)
    ri = find_atom_position(L1f,atomI)
    L1f.commands_string(f'''
        write_data {nebI}
        ''')
    
    if me==0:
        #now write to the info file
        with open(infofile,'a') as f:
            f.write(f"pcsv_{fileIdent}_iPos [{icoord[0]},{icoord[1]},{icoord[2]}]\n")
            f.write(f"pcsv_{fileIdent}_fPos [{fcoord[0]},{fcoord[1]},{fcoord[2]}]\n")
            f.write(f"pcsv_{fileIdent}_box [[{bbox[0][0]},{bbox[0][1]}],[{bbox[1][0]},{bbox[1][1]}],[{bbox[2][0]},{bbox[2][1]}]]\n")
    
    
    
    #####L2 - create the final NEB data file
    ri = find_atom_position(L2,atomI)
    # print(L2.extract_compute('thermo_pe',0,0)*conv)
    recenter_sim(L2,ri)
    # print(L2.extract_compute('thermo_pe',0,0)*conv)
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
    L2f = get_lammps(f'{outfolder}/logs/PrepNEB-Ff.log')
    init_dump(L2f,reset2,0)
    
    
    xyz=outfolder+f'{fileIdent}-NEBFXYZ.data'
    #now create the lowest energy position data file for NEB.
    L2f.commands_string(f'''
                set atom {atomI} x {rf2[0]} y {rf2[1]} z {rf2[2]}
    ''')
    
    NEB_min(L2f)
    
    L2f.commands_string(f'''       
                run 0
                write_data {xyz}
    ''')
    
    
    
    rf2 = find_atom_position(L2f,atomI)
    

        

    plottitle=f"Potential energy landscape around atom {atomI}"
    redXPts=np.transpose([[0,0]])
    allPts=[['x',redXPts]]
    
    
    print(f'got here 2')
    if skipPES != 1 and me==0:
        print(f'plotting PES - {PESimage}')
        plot_PES(PESimage,allPts,xlist,zlist,elist,plottitle)
    
    
    ####Now clean up the dump file to be the correct format for NEB runs
    if me == 0:## ONLY RUN ON ONE PROCESS
        
        with open(nebF,'w+') as f:
            f.write(f'''1\n{atomI} {rf2[0]} {rf2[1]} {rf2[2]}''')

            
        
    return (r_initial,rf2,xyz)


def prep_neb_boomerang_zap(args):
    """
    Prepare the NEB (Nudged Elastic Band) calculation for a boomerang zap run i.e. take one atom and a nearby atom, delete the nearby atom,
    and use the final location of the deleted atom as the final position of the moving atom. The boomerang version of this does it multiple
    times be fore finally creating the initial and final structures. This was a test run that I was doing to determine if this significantly 
    changed the results of the NEB calculation. Create a log file in the proper folder which passed details of this NEB to the rest of the pipeline.
    @param args - needs to have the following information 
        @param dfile - the input file
        @param atomid - The main atom that will be moved
        @param atomF - The atom that will be deleted and moved into.
        @param repeat - total number of times to boomerang
        @param out - output folder
        @param info - information file that passed the details of the run to the NEB.lmp file and Process-NEB.py 
        @param plot - whether to plot (deprecated)
    @return None
    """
    # currentIfile=datafile=args.dfile
    atomI=args.atomid
    outfolder=args.out
    infofile=args.info

    identifier=f'{args.atomid}'
    #fpos=np.array([args.fposx,args.fposy,args.fposz])
    
    total_run=int(args.repeat)+1
    
    
    
    (ipos,fpos,currentIfile) = prep_neb_zap_single(args)


    #L1 = get_lammps(f'{outfolder}/logs/PrepNEB-If.log')
    
    
    full= outfolder+ f'{identifier}-Full.data'
    
    dumpstep=0
    

    if me==0:
        print(f"ipos{ipos} fpos{fpos}")
    #@TODO make this use neb multi jump function by moving that function to locatio and not bond center
    pos_list=[]
    for i in range(total_run):
        pos_list.append(ipos)
        pos_list.append(fpos)
        
    


    with open(infofile,'a') as f:
        if me==0:
            print(f"Num repeats {total_run-1}")
            f.write(f"boomerang {total_run-1}\n")

        for i in range(total_run):
            pos=pos_list[2*i]
            ident=f"j{i}f"
            fileIdent=f"{identifier}-{ident}"
            # if me==0:
            #     print(f"{i} pos{pos} file:{currentIfile}")
            
            write=True #False if i <= (total_run - 2) else True
            
            (nebI, nebF,nebFfullData) = prep_neb_to_location(args,fileIdent,ident,pos,datafile=currentIfile,write_info=write)#,L1=L1)
            currentIfile=nebFfullData#start the next jump where the last left off
            
            pos=pos_list[2*i+1]
            ident=f"j{i}r"
            fileIdent=f"{identifier}-{ident}"
            
            (nebI, nebF,nebFfullData) = prep_neb_to_location(args,fileIdent,ident,pos,datafile=currentIfile,write_info=write)#,L1=L1)
            currentIfile=nebFfullData#start the next jump where the last left off 

            # if me==0:
            #     f.write(f"neb {i} {atomI} {identifier} {nebI} {nebF} {identifier}-{atomI}.log {atomI}\n"+#bash file is expecting a h atom after log file
            #             f"initial-dat-{i} {nebI}\n"+
            #             f"final-dat-{i} {nebF}\n"
            #             )

def prep_neb_boomerang(args):
    """
    Prepare the NEB (Nudged Elastic Band) calculation for a boomerang run i.e. take one atom and place it in a final location. The boomerang
    version of this does it multiple times before finally creating the initial and final structures. This was a test run that I was doing to
    determine if this significantly changed the results of the NEB calculation. Create a log file in the proper folder which passed details 
    of this NEB to the rest of the pipeline.
    @param args - needs to have the following information 
        @param dfile - the input file
        @param atomid - The main atom that will be moved
        @param fpos - The final location of the atom.
        @param repeat - total number of times to boomerang
        @param out - output folder
        @param info - information file that passed the details of the run to the NEB.lmp file and Process-NEB.py 
        @param plot - whether to plot (deprecated)
    @return None
    """
    currentIfile=datafile=args.dfile
    atomI=args.atomid
    outfolder=args.out
    infofile=args.info

    identifier=f'{args.atomid}'
    fpos=np.array([args.fposx,args.fposy,args.fposz])
    
    total_run=int(args.repeat)+1

    L1 = get_lammps(f'{outfolder}/logs/PrepNEB-If.log')
    
    
    full= outfolder+ f'{identifier}-Full.data'
    
    dumpstep=0
    
    #initilize the data files 
    if datafile.endswith(".dump"):
        LT = get_lammps(f'{outfolder}/logs/PrepNEB-LT.log')
        #do this first initialize to get around reaxff issues(charge stuff)
        init_dump(LT,datafile,dumpstep)
        LT.commands_string(f'''
            write_data {full}
            ''')
        
        init_dat(L1,full)
        
    elif datafile.endswith(".data") or datafile.endswith(".dat"):
        init_dat(L1,datafile)
    else:
        print("File is not a .data or .dump")
        return

    ipos = find_atom_position(L1,atomI)
    if me==0:
        print(f"ipos{ipos} fpos{fpos}")
    #@TODO make this use neb multi jump function by moving that function to locatio and not bond center
    pos_list=[]
    for i in range(total_run):
        pos_list.append(fpos)
        pos_list.append(ipos)
    


    with open(infofile,'a') as f:
        if me==0:
            print(f"Num repeats {total_run-1}")
            f.write(f"boomerang {total_run-1}\n")

    for i in range(total_run):
        pos=pos_list[2*i]
        ident=f"j{i}f"
        fileIdent=f"{identifier}-{ident}"
        # if me==0:
        #     print(f"{i} pos{pos} file:{currentIfile}")
        
        write=False if i <= (total_run - 2) else True
        
        (nebI, nebF,nebFfullData) = prep_neb_to_location(args,fileIdent,ident,pos,datafile=currentIfile,write_info=write,L1=L1)
        currentIfile=nebFfullData#start the next jump where the last left off
        
        pos=pos_list[2*i+1]
        ident=f"j{i}r"
        fileIdent=f"{identifier}-{ident}"
        
        (nebI, nebF,nebFfullData) = prep_neb_to_location(args,fileIdent,ident,pos,datafile=currentIfile,write_info=write,L1=L1)
        currentIfile=nebFfullData#start the next jump where the last left off 

        # if me==0:
        #     f.write(f"neb {i} {atomI} {identifier} {nebI} {nebF} {identifier}-{atomI}.log {atomI}\n"+#bash file is expecting a h atom after log file
        #             f"initial-dat-{i} {nebI}\n"+
        #             f"final-dat-{i} {nebF}\n"
        #             )


def prep_interstitial(args):
    """
    Prepare the NEB (Nudged Elastic Band) calculation for a interstitial run i.e. take one atom and find a nearby interstitial location to
    place it at. Create a log file in the proper folder which passed details of this NEB to the rest of the pipeline.
    Realistically, this is better done in the pair_list creation step of the pipeline, thus this function was not used or updated much
    @param args - needs to have the following information 
        @param dfile - the input file
        @param atomid - The main atom that will be moved
        @param repeat - total number of times to boomerang
        @param out - output folder
        @param info - information file that passed the details of the run to the NEB.lmp file and Process-NEB.py 
        @param plot - whether to plot (deprecated)
    @return None
    """
    import NEBTools_setup as nt
    import warnings
    
    
    warnings.filterwarnings('ignore', message='Setting an item of incompatible dtype is deprecated and will raise in a future error of pandas.*')
    outfolder=args.out
   
    seed=args.atomid
    
    infofile=args.info
    file=args.dfile
    plot=args.plot
    dumpstep=0
    skipPES=1
    
    
    bulk_low_z = 18
    bulk_high_z = 29
    
    # plt.rcParams["figure.autolayout"] = True
    
    L1 = get_lammps(f'{outfolder}/logs/PrepNEB-I.log')
    
    fileIdent=f'{seed}'


    nebI=outfolder+f'{fileIdent}-NEBI.data'
    nebF=outfolder+f'{fileIdent}-NEBF.data'
    full= outfolder+ f'{fileIdent}-Full.data'
    
    # PESimage=outfolder+f"PES({fileIdent}).png"
    # ovitoFig=outfolder+f"{fileIdent}-Ovito.png"

    
    #initilize the data files 
    if file.endswith(".dump"):
        LT = get_lammps(f'{outfolder}/logs/PrepNEB-LT.log')
        #do this first initialize to get around reaxff issues(charge stuff I think)
        init_dump(LT,file,dumpstep)
        LT.commands_string(f'''
            write_data {full}
            ''')
        #
        init_dat(L1,full)
        # init_dat(L2,full)
        
    elif file.endswith(".data") or file.endswith(".dat"):
        init_dat(L1,file)
        # init_dat(L2,file)
    else:
        print("File is not a .data or .dump")
        return
    
    if me==0:
        print(f'seed - {seed}')
    
    pre_atoms=L1.get_natoms()
    
    # try:
    #      find_atom_position(L1,pre_atoms+1)
    # except Exception as e:
    #     print(e)
    
    tempfile_name1=f'{fileIdent}-pre_addedO.data'
    tempfile1=outfolder+f'{tempfile_name1}'
    
    L1.commands_string(f'''
        write_data {tempfile1}
        ''')

    place_random_O(L1,[bulk_low_z,bulk_high_z],seed)
    
    atomI=L1.get_natoms()
    
    if me==0:
        print(f"pre {pre_atoms} post {atomI}")
    
    
    
    NEB_min(L1)
    
    ##### L1 - create the initial NEB data file
    ri = find_atom_position(L1,atomI)


    tempfile_name=f'{fileIdent}-addedO.data'
    tempfile=outfolder+f'{tempfile_name}'
    L1.commands_string(f'''
        write_data {tempfile}
        ''')
    if me==0:
        print('here1')
    (atoms, simbox) = nt.read_file_data_bonds(outfolder,tempfile_name)
    if me==0:
        print('here2')
        
    sibc=nt.find_neighboring_sibc(atoms,atomI)
    
    if len(sibc) == 0:
        print('No suitible bond centers found?')
        return
    
    if me==0:
        print('here3')
        
    args.atomid=atomI
    args.dfile=tempfile
    
    
    prep_neb_to_bond_center(args,seed,atomF1=sibc[0][0],atomF2=sibc[0][1])
    
    # # if me==0:
    # #     #write info file
    # #     with open(infofile,'a') as f:
    # #         f.write(f"zap\n")
    # #         f.write(f"neb 0 {atomI} {fileIdent} {nebI} {nebF} {fileIdent}.log {atomI}\n"+#bash file is expecting a h atom after log file
    # #                 f"initial-dat-0 {nebI}\n"+
    # #                 f"final-dat-0 {nebF}\n"
    # #                 )
    
    
    # icoord=ri#have to use the coordinates before recentering
    # #print(L1.extract_compute('thermo_pe',0,0)*conv)
    # bbox=recenter_sim(L1,ri)
    # #print(L1.extract_compute('thermo_pe',0,0)*conv)

    # #ri = find_atom_position(L1,atomI)
    # NEB_min(L1)
    # r_initial = find_atom_position(L1,atomI)
    
    # L1.commands_string(f'''
    # write_data {nebI}
    # ''')



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--style')
    parser.add_argument('--out')
    parser.add_argument('--etol')
    parser.add_argument('--ts')
    parser.add_argument('--dfile')
    parser.add_argument('--atomid',type=int)
    parser.add_argument('--zapid')
    parser.add_argument('--plot',type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate nice mode.")
    parser.add_argument('--bclist')
    parser.add_argument('--info')
    parser.add_argument('--fposx')
    parser.add_argument('--fposy')
    parser.add_argument('--fposz')
    parser.add_argument('--h2x')
    parser.add_argument('--h2y')
    parser.add_argument('--h2z')
    parser.add_argument('--bc1')
    parser.add_argument('--bc2')
    parser.add_argument('--repeat')
    
    
    
    args=parser.parse_args()
    
    #define variables for this run
    dt=args.ts
    etol=args.etol
    a=5.43
    skipPES=0
    
    me = MPI.COMM_WORLD.Get_rank()
    comm=MPI.COMM_WORLD.Get_size()

    
    
    
    #conversion from kcal/mol to eV
    conv=0.043361254529175


    
    
    
    # unit testing
    # xpos=float(sys.argv[8])
    # ypos=float(sys.argv[9])
    # zpos=float(sys.argv[10])
    # fpos=[xpos,ypos,zpos]
    # prep_neb_to_location(datafile,dumpstep,atomI,fpos,fileIdent,ident,outfolder,infofile)
    
    
    # zapID=int(sys.argv[8])
    # hID=int(sys.argv[9])
    # nebfiles=prep_neb_zap_multi(datafile,dumpstep,atomI,[hID],zapID,outfolder,infofile,plot)
    # print(f"{atomI}-{hID} zapping {zapID}")
    
    
    # zapID=int(sys.argv[8])
    # prep_neb_zap_single(datafile,dumpstep,atomI,zapID,outfolder,infofile,plot)

    #print(f"{outfolder} {etol} {dt} {datafile} {atomI} {numjumps} {bondcenter_list}")
    if args.style=='multi_jump':
        nebFiles =prep_neb_multi_jump(args)#,datafile,dumpstep,atomI,outfolder,infofile,plot)
    elif args.style=='single_jump':
        fileIdent=f'{args.atomid}'
        ident=fileIdent
        fpos=np.array([args.fposx,args.fposy,args.fposz])
        
        prep_neb_to_location(args,fileIdent,ident,fpos,datafile=None,L1=None,dumpstep=0)
    elif args.style=='h_to_h2':
        fileIdent=f'{args.atomid}'
        ident=fileIdent
        fpos=np.array([args.fposx,args.fposy,args.fposz])
        # h2pos=np.array([args.h2x,args.h2y,args.h2z])
        prep_neb_h_to_h2_region(args,fileIdent,ident,fpos,datafile=None,L1=None,dumpstep=0)
    elif args.style=='single_zap':
        args.zapid=int(args.zapid)
        prep_neb_zap_single(args)
    elif args.style=='boomerang':
        
        prep_neb_boomerang(args)
        
        # repeats=int(args.repeat)
        # fileIdent=f'{args.atomid}'
        # ident=fileIdent
        # fpos=np.array([args.fposx,args.fposy,args.fposz])
        # prep_neb_to_location_boomerang(args,fileIdent,ident,fpos,datafile=None,L1=None,dumpstep=0,boomerang=repeats)
        
    elif args.style=='boomerang_zap':
        prep_neb_boomerang_zap(args)
        
    elif args.style =='interstitial':
        prep_interstitial(args)
    
    MPI.Finalize()
    exit()