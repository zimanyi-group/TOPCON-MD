#!/usr/bin/env python
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from sympy import Point3D, Line3D, Plane

import numpy as np
import statistics
from scipy import stats
from pathlib import Path
from lammps import lammps

# import dask.dataframe as dd
# from dask.multiprocessing import get
# from operator import itemgetter
# from math import degrees
import pandas as pd
pd.options.mode.chained_assignment = None 
from ast import literal_eval
import os
from mpi4py import MPI
import time
# import matplotlib.cm as cm
# import matplotlib.colors as colors
from ase.geometry import get_angles
import ase.cell

from matplotlib.ticker import MaxNLocator
from pandas.api.types import is_numeric_dtype
from numpy.linalg import norm

jp=0


me = MPI.COMM_WORLD.Get_rank()
numproc=MPI.COMM_WORLD.Get_size()


conv=0.043361254529175

SiOBondOrder=0.9
image_folder="/home/agoga/documents/code/topcon-md/neb-out/analysis-images/"
datafolder="/home/agoga/documents/code/topcon-md/data/neb/"

v=4.74e-20
w=1.6e-7
HNumToConcentration=w/v

def pbc_midpoint(simbox,p1,p2):
    ret=[0,0,0]
    for i, (a, b) in enumerate(zip(p1, p2)):
        delta = abs(b - a)
        dimension=simbox[i,1]-simbox[i,0]
        
        if delta > dimension/2:
            if b>a:
                b+=dimension
            else:
                a+=dimension
            
        ret[i]=(b+a)/2
        
    return ret

def pbc_vec_subtract(simbox, posi,posf):
    ret=[0,0,0]
    for i, (a, b) in enumerate(zip(posi, posf)):
        delta = abs(b - a)
        dimension=simbox[i,1]-simbox[i,0]
        if delta > dimension/2:
            if b < a:
                b+=dimension
            else:
                a+=dimension
            
        ret[i]=b-a
        
    return ret

def pbc_dist(simbox, pos1,pos2):
    total = 0
    for i, (a, b) in enumerate(zip(pos1, pos2)):
        delta = abs(b - a)
        dimension=simbox[i,1]-simbox[i,0]
        if delta > dimension- delta:
            delta = dimension - delta
        total += delta ** 2
    return total ** 0.5


def vec_projection(v1,v2):
    return np.dot(v1,v2)/np.linalg.norm(v2)

def vec_proj_to_plane(v1,v2):
    s_proj=vec_projection(v1,v2)
    return v1-s_proj*v2
    

def pbc_dist_point_to_vec(simbox, p1,p2,distPoint):
    
    if np.array_equal(p1,p2) or np.array_equal(p1,distPoint) or np.array_equal(distPoint,p2) :
        return None
    vec1=pbc_vec_subtract(simbox,p1,p2)
    vec2=pbc_vec_subtract(simbox,distPoint,p1)
    try:
        dist = norm(np.cross(vec1,vec2))/norm(vec1)
    except:
        print(f"pbc_dist_point_to_vec failed p1: {p1} p2: {p2} dp: {distPoint}")
        
    #print(f"pbc_dist_point_to_vec - {distPoint} to {p2}-{p1}={dist}")
    return dist


def apply_point_vec_dist(df,simbox,p1,p2,atomtype=None,col='pos'):
    pvdcol='pointvecDist'
    if df.empty:
        return df
    distdf=df.copy()
    
    if atomtype is not None:
        distdf=distdf[distdf["type"]==atomtype]

    distdf[pvdcol]=distdf.apply(lambda row: pbc_dist_point_to_vec(simbox,p1,p2,row[col]),axis=1)

    return (distdf,pvdcol)
           

def apply_dist_from_pos(df,simbox,pos,atomtype=None,col='pos'):
    distdf=df.copy()
    #pos=df.at[atom,'pos']
    
    if atomtype is not None:
        distdf=distdf[distdf["type"]==atomtype]
    #partdf=dd.from_pandas(distdf,npartitions=numproc)
    # print(distdf.iloc[0].to_string())
    #.swifter.progress_bar(False).allow_dask_on_strings(enable=True).apply
    distdf['dist']=distdf.apply(lambda row: pbc_dist(simbox,row[col],pos),axis=1)
    # print(distdf.iloc[0].to_string())
    #distdf['dist']=partdf.map_partitions(lambda df: df.apply((lambda row: pbc_dist(simbox,row['pos'],pos)),axis=1),meta=('pos', object)).compute(scheduler='threads')

    #distdf.drop(index=atom,inplace=True,errors='ignore')
    return distdf #for now just return the number of atoms

def unit_vector(vector):
    try:
        ret=vector / np.linalg.norm(vector)
        return ret
    except:
        print('got')
        print(vector)

def stats_from_csv_name(csvname):
    filename=csvname.removesuffix('.dat').removesuffix('.data').removesuffix('.dump').removesuffix('.csv')
    ratio=filename.split('-')[0]
    Hnum=filename.split('-')[1]
    #print(f"ratio: {ratio}, Hnum:{Hnum}")
    return (ratio,Hnum)

def angle_between_vec(box,v1,v2,debug=False):
    cell=ase.cell.Cell([[box[0,0],box[1,1],box[2,1]],[box[0,1],box[1,0],box[2,1]],[box[0,1],box[1,1],box[2,0]]])
    ang=get_angles([v1],[v2],cell,True)

    return (ang,[v1,v2])


def angle_between_pts(box,p1, p2,pm,debug=False):
    # if v1 is None or v2 is None:
    #     return 2*np.pi

  
    cell=ase.cell.Cell([[box[0,0],box[1,1],box[2,1]],[box[0,1],box[1,0],box[2,1]],[box[0,1],box[1,1],box[2,0]]])

            
    # print(v1.dtype)
    # print(f"{v1} {v2} {cell}")
    
    
    print(f"--- angle_between_pts ---\np1{p1} p2{p2}\npm{pm}\nbox{box}") if debug else None
    
    lenlist=[(box[0,1]-box[0,0]),(box[1,1]-box[1,0]),(box[2,1]-box[2,0])]
    #@TODO fix this for periodic boundary conditions, any atoms that have one or more over an  edge will produce low angles
    imiddle=pm.copy()
    fmiddle=pm.copy()
    for k in range(len(p1)):#go through each dimension
        
        if abs(p1[k]-imiddle[k]) > lenlist[k]/2:
            
            if p1[k]<imiddle[k]:
                p1[k]=p1[k]+lenlist[k]
                print(f"{k}th dimension (a)") if debug else None
            else:
                imiddle[k]=imiddle[k]+lenlist[k]
                print(f"{k}th dimension (b)") if debug else None
                
        if abs(p2[k]-fmiddle[k]) > lenlist[k]/2:
            if p2[k]<fmiddle[k]:
                p2[k]=p2[k]+lenlist[k]
                print(f"{k}th dimension (c)") if debug else None
            else:
                fmiddle[k]=fmiddle[k]+lenlist[k]
                print(f"{k}th dimension (d)") if debug else None
                
    
    v1=p1-imiddle
    v2=p2-fmiddle
    
    
    
    
    ang=angle_between_vec(box,v1,v2)

    print(f"--- post fixing ---\np1{p1} p2{p2}\nimiddle{imiddle} fmiddle{fmiddle}\nbox{box}\nang={ang}") if debug else None
    return (ang,[v1,v2])

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

def NEB_min(L,etol):
    L.commands_string(f'''minimize {etol} {etol} 10000 10000''')

def find_bond_preference(box,file,atom,midpoint,sepvec):
    from sympy.abc import u,v
    
    
    
    ti=time.time()
    #Need two lammps instances so that when removing an atom and minimizing we don't increase time for final NEB image minimization
    L = lammps('mpi',["-log",'none',"-screen",'none'])
    init_dat(L,file)
    
    L.commands_string(f'''
        variable x{atom} equal x[{atom}]
        variable y{atom} equal y[{atom}]
        variable z{atom} equal z[{atom}]
        ''')
    
    x = L.extract_variable(f'x{atom}')
    y = L.extract_variable(f'y{atom}')
    z = L.extract_variable(f'z{atom}')
    orig=np.array([x,y,z])
    #(df, box) = read_data(file)

    
    dplane=Plane(Point3D(midpoint),normal_vector=sepvec)
    parametrizedpt=dplane.arbitrary_point(u,v)
    zpoint_plane=np.array(parametrizedpt.subs({u:np.cos(0),v:np.sin(0)}).evalf(),dtype=float)-midpoint
    
    ang=angle_in_plane(box,midpoint,sepvec,orig,zpoint_plane)
    
    rtot=2
    angtot=8
    twopi=2*np.pi
    tstep=0
    tstart=ang
    radius=0.04
    pts=[]
    rmin=0.3
    rmax=0.6
    parametrizedpt=dplane.arbitrary_point(u,v)

    newpt=parametrizedpt.subs({u:radius*np.cos(ang),v:radius*np.sin(ang)}).evalf()
    mpt=np.array(newpt)
    
    #     for offset in [0]: #[-0.01,0,0.01]:
    #     for radius in np.linspace(rmin,rmax,rtot):
    #         for theta in np.linspace(tstart,tstart+twopi/2,angtot,endpoint=False):
    #             newpt=parametrizedpt.subs({u:radius*np.cos(theta),v:radius*np.sin(theta)}).evalf()
    #             newpt.translate(0,offset,0)
    #             pts.append([newpt,theta,radius])
    #         tstart+=tstep



    # tot=len(pts)
    # xi, yi, zi = midpoint[0], midpoint[1], midpoint[2]
    # L.commands_string(f'''
    #         set atom {atom} x {xi} y {yi} z {zi}
    #         ''')
    # s=0
    # Ef=0
    # Ei = L.extract_compute('thermo_pe',0,0)*conv
    # minE=0
    # minpt=[midpoint,0,0]
    # for ptt in pts:
    #     pt=ptt[0]
    #     xf=pt[0]
    #     yf=pt[1]
    #     zf=pt[2]
        
    #     L.commands_string(f'''
    #         set atom {atom} x {xf} y {yf} z {zf}
    #         run 0
    #         ''')
        
    #     s+=1
    #     Ef = L.extract_compute('thermo_pe',0,0)*conv
        
    #     if me==0:
    #         print(f"Step {s}/{tot}: {Ef - Ei} ang:{ptt[1]} r:{ptt[2]}")
            
    #     if Ef < minE:
    #         minE=Ef
    #         minpt=ptt

    # L.commands_string(f'''
    #         set atom {atom} x {minpt[0][0]} y {minpt[0][1]} z {minpt[0][2]}
    #         run 0
    #         ''')

    # x = L.extract_variable(f'x{atom}')
    # y = L.extract_variable(f'y{atom}')
    # z = L.extract_variable(f'z{atom}')
    # mpt=[x,y,z]
    # initialpos=[xi,yi,zi]
    # tf=time.time()
    # if me==0:
    #     print(f"Atom {atom} from {initialpos} to {mpt}, ang:{minpt[1]}, r:{minpt[2]}\n{Ei}  {minE} in {tf-ti}s")
        
    return mpt

def angle_in_plane(box,plane_point,plane_vec,orig_point,plane_zero_vec,ax=None):
    # from sympy.abc import g,f
    # dplane=Plane(Point3D(plane_point),normal_vector=plane_vec)
    plane_point=np.array(plane_point,dtype=float)
    plane_vec=np.array(plane_vec,dtype=float)
    orig_point=np.array(orig_point,dtype=float)
    orig_to_plane=pbc_vec_subtract(box,plane_point,orig_point)
    # parametrizedpt=dplane.arbitrary_point(g,f)
    # zpoint_plane=np.array(parametrizedpt.subs({g:np.cos(0),f:np.sin(0)}).evalf(),dtype=float)
    # plane_zero_vec=zpoint_plane



    orig_plane_proj=vec_proj_to_plane(orig_to_plane,plane_vec)
    if all(t==0 for t in orig_plane_proj):
        print("projection is [0,0,0]")
        return 0

    opp_vec=orig_plane_proj/np.linalg.norm(orig_plane_proj)


    #find the angle between the two vecotrs IN THE PLANE
    # n=sepvec/np.linalg.norm(sepvec)
    angdeg=angle_between_vec(box,opp_vec,plane_zero_vec)[0][0]
    anginplane_orig=np.radians(angdeg)#-np.dot(n,np.cross(opp_vec,plane_zero_vec))
    #print(f"rad: {anginplane_orig} deg:{angdeg}")
    dmat=[plane_zero_vec,plane_point+opp_vec,plane_vec]
    sign=np.linalg.det(dmat)
    anginplane_orig=anginplane_orig if sign > 0 else -anginplane_orig
    
    if ax is not None:
        opp_pt=plane_point+orig_plane_proj
        plto=np.array([plane_point,opp_pt]).T
        ax.plot(plto[0],plto[1],plto[2],color='r')
        
        plto=np.array([orig_point,opp_pt]).T
        ax.plot(plto[0],plto[1],plto[2],color='r')

        # zv=np.array([plane_point,plane_zero_vec]).T
        # ax.plot(zv[0],zv[1],zv[2],color='blue')
        
    return anginplane_orig

def find_local_minima_position(file,atom,initial_guess):
    ti=time.time()
    #Need two lammps instances so that when removing an atom and minimizing we don't increase time for final NEB image minimization
    L = lammps('mpi',["-log",'none',"-screen",'none'])
    init_dat(L,file)

    len=1
    perDim=10
    step=len/perDim
    elist=np.zeros([perDim,perDim,perDim])

    dlist=np.arange(-len/2,len/2,step)
    
    tot = perDim**3
    s=1
    
    
    xi, yi, zi = initial_guess[0], initial_guess[1], initial_guess[2]
    minx, miny, minz = xi, yi, zi
    
    L.commands_string(f'''
                    set atom {atom} x {xi} y {yi} z {zi}
                    run 0
                    ''')

    Ef=0
    Ei = L.extract_compute('thermo_pe',0,0)*conv
    minE=Ei
    for i in range(perDim):
        for j in range(perDim):
            for k in range(perDim):
            
                x=dlist[i]
                y=dlist[j]
                z=dlist[k]

                
                xf = xi + x
                yf = yi + y
                zf = zi + z
                
                
                
                L.commands_string(f'''
                    set atom {atom} x {xf} y {yf} z {zf}
                    run 0
                    ''')
                s+=1
                Ef = L.extract_compute('thermo_pe',0,0)*conv
                # elist[i,j,k]=Ef
                
                if me==0:
                    print(f"Step {s}/{tot}: {Ef - minE}")
                    
                if Ef < minE:
                    minx=xf
                    miny=yf
                    minz=zf
                    
                    
                    # if me==0:
                    #     print(f"New minima; {minE} to {Ef}")
                        
                    minE=Ef
                    
                
    
    lowestpos=[minx,miny,minz]
    initialpos=[xi,yi,zi]
    tf=time.time()
    if me==0:
        print(f"Atom {atom} from {initialpos} to {lowestpos}: {Ei}  {minE} in {tf-ti}")
        
    return lowestpos

def angle_between_df(df, col, refang=[0,0,1]):
    
    for index, row in df.iterrows():
        ipos = np.array(row["iPos"])
        fpos = np.array(row["fPos"])
        box = np.array(row["box"])
        
        cell=ase.cell.Cell([[box[0,0],box[1,1],box[2,1]],[box[0,1],box[1,0],box[2,1]],[box[0,1],box[1,1],box[2,0]]])

            
        # print(v1.dtype)
        # print(f"{v1} {v2} {cell}")
        
        
        lenlist=[(box[0,1]-box[0,0]),(box[1,1]-box[1,0]),(box[2,1]-box[2,0])]
        #@TODO fix this for periodic boundary conditions, any atoms that have one or more over an  edge will produce low angles
        for k in range(len(ipos)):#go through each dimension
            if abs(ipos[k]-fpos[k]) > lenlist[k]/2:
                if ipos[k]<fpos[k]:
                    ipos[k]=ipos[k]+lenlist[k]
                else:
                    fpos[k]=fpos[k]+lenlist[k]

        v1=ipos-fpos
        ang=get_angles([v1],[refang],cell,True)
        
        df.at[index,col]=ang#angle_between(fpos-ipos,refang)
    return df

def create_bond_file(datapath, file,bondfile):
    L= lammps('mpi',["-log",f'{datapath}/scratchfolder/CreateBonds.log',"-screen",'none'])
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

        timestep 0.5

        variable printevery equal 100
        variable restartevery equal 0
        variable datapath string "data/"


        variable massSi equal 28.0855 #Si
        variable massO equal 15.9991 #O
        variable massH equal  1.00784 #H
        

        read_data {datapath+file}
        
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
        fix f1 all reaxff/bonds 1 {datapath}/scratchfolder/{bondfile}
        
        run 0
        ''')
    
def read_bonds(df,file):
    # alist=[]
    #create new columns in the datafram
    with open(file,'r') as f:

        for line in f:
            #skip the first 7 lines for header
            ainfo=[]
            l=line.split()
            if l[0] == '#':
                continue
            i=0#the current item in the line we're on

            id=int(l[0])
            typ=int(l[1])
            nb=int(l[2])
            if typ == 1:
                typ='Si'
            elif typ==2:
                typ='O'
            elif typ==3:
                typ='H'
            if df.at[id,'type']!=typ:
                print(f'Big bombad making bonds - Atom{id} doesnt match types')
                
            df.at[id,'nb']=nb
            i=3
            blist=[]
           
           #run through each bond and grab the id and bond order(index offset by the number of bonds)
            for j in range(nb):
                blist.append([int(l[i+j]),float(l[i+j+nb+1])])
            #print(blist)
            df.at[id,"bonds"]=""
            df.at[id,"bonds"]=blist

    return df

def read_data(file):
    dlist=[]
    skipindex=0
    simbox=np.zeros((3,2))

    with open(file,'r') as f:

        
        for line in f:
            l=line.split()
            
            if "xlo" in line:
                simbox[0,0]=l[0]
                simbox[0,1]=l[1]
            if "ylo" in line:
                simbox[1,0]=l[0]
                simbox[1,1]=l[1]
            if "zlo" in line:
                simbox[2,0]=l[0]
                simbox[2,1]=l[1]
            #skip the first 17 lines for header
            if skipindex<17:
                skipindex+=1
                continue
            
            dinfo=[]
            
            if len(l)<6:
                continue
            #example line 
            #1 1 0.03793783944547108 1.2804479390923988 1.2230759130866515 1.8085851741085164 0 0 0
            id=int(l[0])
            typ=int(l[1])
            if typ == 1:
                typ='Si'
            elif typ==2:
                typ='O'
            elif typ==3:
                typ='H'
                
            q=float(l[2])#charge
            x=float(l[3])
            y=float(l[4])
            z=float(l[5])
            pos=np.array([x,y,z])
            dinfo=[id,typ,q,x,y,z,pos]
            
            #add this row to the dataframe
            dlist.append(dinfo)

    df = pd.DataFrame(dlist,columns=['id','type','q','x','y','z','pos'])
    df.set_index('id',inplace=True,drop=True)

    return (df,simbox)

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

        timestep 0.5

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


#@TODO rename plot bond investigation
def temp(path,dfile):
    (atoms,box) = read_file_data_bonds(path,dfile)
    
    pinholeCenter=[27,27,20]
    
    siatoms=atoms[atoms['type']=='Si']

    osiobonds=[]

    for index,row in siatoms.iterrows():
        bonds=row['bonds']
        curpos=row['pos']
        numbonds=len(bonds)
        
        pinholeCenterCurZ=pinholeCenter
        pinholeCenterCurZ[2]=curpos[2]
        
        dist=pbc_dist(box,pinholeCenterCurZ,curpos)
        if dist > 13:
            continue
        
        
        for ne in bonds:
            ni=ne[0]
            neitype=atoms.at[ni,'type']
            
            if neitype=='O' and ne[1]>SiOBondOrder:
                
                sibonds=atoms.at[ni,'bonds']
                        
                for nnn in sibonds:
                    nnni=nnn[0]
                    nnntype=atoms.at[nnni,'type']
                    if nnntype=='Si'and nnni!=index and nnn[1]>SiOBondOrder:
                        osiobonds.append([index,nnni,ni,numbonds])
                        
            
    distlist=[]
    anglist=[]
    bondslist=[]
    difflist=[]
    ablist=[]
    for ob in osiobonds:
        si1=ob[0]
        si2=ob[1]
        oatom=ob[2]
        
        spos1=atoms.at[si1,'pos']    
        spos2=atoms.at[si2,'pos']   
        opos=atoms.at[oatom,'pos']
        
        d1=abs(np.linalg.norm(pbc_vec_subtract(box,opos,spos1)))
        d2=abs(np.linalg.norm(pbc_vec_subtract(box,opos,spos2)))
        
        dis= pbc_dist_point_to_vec(box,spos1,spos2,opos)
        ang=angle_between_pts(box,spos1,spos2,opos)[0][0]
        if dis is not None:
            distlist.append(dis)
            anglist.append(ang)
            bondslist.append(ob[3])
            difflist.append(abs(d1-d2))
        
    # print(distlist)
    # print(anglist)
    # plt.scatter(bondslist,distlist)
    # plt.show()
    
    for nb in range(1):
        angblist=[]
        for i in range(len(bondslist)):
            #if bondslist[i]==nb:
            angblist.append(distlist[i])

        if len(angblist)==0:
            continue
        
        plt.hist(angblist,bins=10)
        plt.title(f"Distance from Oxygen atom to its bonded Si's seperation vector")
        plt.xlabel("Distance(Angstroms)")
        plt.ylabel("Counts")
        plt.show()
        # avg=np.mean(difflist)
    # print(avg)

def read_file_data_bonds(datapath,dfile):

    (dfdata,simbox)=read_data(datapath+dfile)
    filename=dfile.removesuffix('.dat').removesuffix('.data').removesuffix('.dump')
    bondfile=filename+".bonds"
    os.makedirs(datapath+'/scratchfolder/', exist_ok=True)
    create_bond_file(datapath,dfile,bondfile)

    df=read_bonds(dfdata,datapath+'/scratchfolder/'+bondfile)

    return (df,simbox)


def df_combine_H(df,spread=5):
    df=df.sort_values(["ratio","Hnum"])

    hlist=np.array(df['Hnum'].unique()).astype(int)

    
    done=[]
    for h in hlist:
        if h in done:
            continue
        for ho in hlist:
            if h==ho :
                continue
            
            if  ho > h - spread and ho < h + spread:
                print(f"{h} and {ho}")
                df.loc[df["Hnum"]==str(ho),"Hnum"]=str(h)

                done.append(h)
                done.append(ho)
    return df


def load_data_from_csv(csvname):
    global datafolder
    datafile=csvname.removesuffix('.dat').removesuffix('.data').removesuffix('.dump').removesuffix('.csv')+'.dat'
    return read_data(datafolder+datafile)

def load_data_and_bonds_from_csv(csvname):
    global datafolder
    datafile=csvname.removesuffix('.dat').removesuffix('.data').removesuffix('.dump').removesuffix('.csv')+'.dat'
    return read_file_data_bonds(datafolder,datafile)

def mean_str(col):
    global jp
    if is_numeric_dtype(col):
        return col.mean()
    else:
        if col.nunique()>1 and jp <10:
            print(f"Non unique column in clean_csvs functions {col.unique()}")
            jp+=1
        return col.unique()[0]

def clean_csvs(csvlist,finalpath):
    dflist=[]
    numcsv=len(csvlist)
    print(numcsv)
    for csvpath in csvlist:
        df = pd.read_csv(csvpath)
        csvn=csvpath.split('/')[-1]
        df["csvname"]=""
        df=df.assign(csvname=csvn)
        dflist.append(df)

    if len(dflist)>1:
        combodf=pd.concat(dflist,ignore_index=True)
    else:
        combodf=dflist[0]


    #average over repeated runs, so any repeated pairs in a specific csv
    grpdf=combodf.groupby(["csvname","pair"],axis=0).agg(mean_str).reset_index()#

    
    ucsv=grpdf["csvname"].unique()
    print(f"{len(combodf)} rows before merging pair's and {len(grpdf)} after")
    print(f"Took {numcsv} csv's down to {len(ucsv)}.")
    for c in ucsv:
        cdf =grpdf[grpdf["csvname"]==c]
        
        cdf.to_csv(finalpath+c)
    
    return grpdf
    
    
def clean_pairfiles(df,pathtopairs):
    ucsvs=df["csvname"].unique()
    
    for csv in ucsvs:
        csvdf=df[df["csvname"]==csv]
        upairs=csvdf["pair"].unique()
        pfile=pathtopairs+csv.removesuffix(".csv")+"-pairlist.txt"
        if os.path.exists(pfile):
            with open(pfile,"r+") as f:
                lines=f.readlines()
                lines.reverse()
                f.seek(0)
                done=0
                notdone=0
                for l in lines:
                    #print(l)
                    pair=l.split()[0]+"-"+l.split()[1]
                    if pair in upairs:
                        done+=1
                    else:
                        notdone+=1
                        f.write(l)
                print(f"{done} done and {notdone} not done")
                f.truncate()
        else:
            print(f"{pfile} doesn't exist.")
        
        
def csv_to_df(csvpath,includebad=False):
    csvname=csvpath.split('/')[-1]
    #print(csvname)
    (cratio,cHnum)=stats_from_csv_name(csvname)
    df=pd.read_csv(csvpath)
    
    
    if not includebad and "fail" in df.columns:
        df=df[df["fail"]!=True]
        
    df['ratio']=""
    df['Hnum']=""
    df['csvname']=""
    df=df.assign(ratio=cratio)#assign all rows in the ratio column
    df=df.assign(Hnum=cHnum)
    df=df.assign(csvname=csvname)
    cols=["iPos","fPos","box"]
    
    if "iPos" in df.columns:
        df['iPos']=df.iPos.apply(literal_eval)
        #df['iPos']=df.iPos.apply(lambda x: [literal_eval(x)])#literal_eval)
    if "fPos" in df.columns:
        df['fPos']=df.fPos.apply(literal_eval)
    if "box" in df.columns:
        df['box']=df.box.apply(literal_eval)
    
    df=df.sort_values(by=['ratio'])
    return df



def csvs_to_df(csvlist):
    dflist=[]
    for csvpath in csvlist:
        df = csv_to_df(csvpath)
        dflist.append(df)


    combodf=pd.concat(dflist,ignore_index=True)
    return combodf


def find_movers_neighbor(df,curatom,zappdatom,natom="Si",nnatom="O"):
    curbonds=df.at[curatom,'bonds']

    on=[]
    sin=[]
    checkatom=None
    
    for n in curbonds:
        ni=n[0]
        bo=n[1]

        neitype=df.at[ni,'type']
        if neitype ==natom:
            neibonds = df.at[ni,'bonds']
        
            for nn in neibonds:
                nni=nn[0]
                nnbo=nn[1]

                #found the interested atom
                if nni == zappdatom:
                    return ni
    return None
                
def find_suitable_neighbors(df,curatom,zappdatom,natom="Si",nnatom="O"):
    
    ni=find_movers_neighbor(df,curatom,zappdatom,natom,nnatom)
    if ni is None:
        return 0
    
    curbonds=df.at[curatom,'bonds']
    neibonds = df.at[ni,'bonds']
    for b in curbonds:
        if b[0] == ni:
            return b[1]


def find_nnneighbor(df,curatom,zappdatom,natom="Si",nnatom="O"):
    curbonds=df.at[curatom,'bonds']


    for n in curbonds:
        ni=n[0]
        bo=n[1]

        neitype=df.at[ni,'type']
        if neitype ==natom:
            neibonds = df.at[ni,'bonds']
        
            for nn in neibonds:
                nni=nn[0]
                nnbo=nn[1]

                #found the interested atom
                if nni == zappdatom:
                    return ni
    return None


def angle_between_pts_df(df,valcol,posdf=None,box=None):
    #this function finds the angle between pts of NEB vacancy pairs
    
    #df=basedf.copy()
    csvlist=df["csvname"].unique()
    
    for csvfile in csvlist:
        dfc=df[df["csvname"]==csvfile]
        (ratio,Hnum)=stats_from_csv_name(csvfile)
        
        if posdf == None and box == None:
            (posdf,box)=load_data_and_bonds_from_csv(csvfile)
        #print(posdf.to_string())
        for index, row in dfc.iterrows():
            pair=row['pair']
            ipos=row['iPos']
            fpos=row['fPos']
            box=np.array(row['box'])
            mover=int(pair.split('-')[0])
            zapped=int(pair.split('-')[1])
            # print(csvfile)
            # print(mover)
            # print(zapped)
            fsn=find_movers_neighbor(posdf,mover,zapped)
            # print(fsn)
            middle=posdf.at[fsn,"pos"]
            
            debug = False
            # if pair=="4704-5380":
            #     debug = True
                
            (ang, ret) = angle_between_pts(box,ipos,fpos,middle,debug)
            
            # if ang <20:
            #     print(f"fudge and crackers {pair}")
            #     print(f"box:{box}\ni:{row['iPos']} f:{row['fPos']}\n m: {middle}\n -- v1:{ret[0]} v2:{ret[1]}")

            df.at[index,valcol]=ang[0]
        
        print(f'angle_between_pts_df done with {csvfile}')
    return df


def find_final_si_pair(bonddf,simbox,mover,f_loc=None):
    si_atoms=[]
    
    # if zapped is not None: 
    #     #this is a zap style run
    #     zappedbonds=bonddf.at[zapped,'bonds']

    #     for n in zappedbonds:
    #         ni=n[0]
    #         bo=n[1]

    #         neitype=bonddf.at[ni,'type']
    #         if neitype =="Si":
    #             si_atoms.append(ni)
                
    # elif f_loc is not None:
    radius=1.75#ang
    #this is a move to location style
    distdf=apply_dist_from_pos(bonddf,simbox,f_loc,"Si")
    distdf=distdf[distdf['dist']<radius]

    for i,r in distdf.iterrows():
        si_atoms.append(i)
        
    if len(si_atoms) != 2:
        print(f'Found {len(si_atoms)} Si neighbors for location {f_loc}')
        return []
        
    final_pair=None#[si_atoms[0],si_atoms[1]]
    
    moverbonds=bonddf.at[mover,'bonds']
    
    #now find the Si that is attached to the mover and put it first in the pair list

    for n in moverbonds:
        ni=n[0]
        bo=n[1]

        neitype=bonddf.at[ni,'type']
        if neitype =="Si":
            if ni == si_atoms[0]:
                final_pair=[si_atoms[0],si_atoms[1]]
                #print(f"For {mover} pair set as {final_pair}")
                pt=1
            if ni == si_atoms[1]:
                final_pair=[si_atoms[1],si_atoms[0]]
                #print(f"For {mover} pair set as {final_pair}")
                pt=1
   
    if final_pair is None:
        print('Issues in find_final_si_pair')
    #final_pair=np.array(final_pair,dtype=int)  
    return final_pair
                
    
def feb_final_sort(entry1,entry2):
    if entry1[2]>entry2[2]:
        return 1
    elif entry1[2]<entry2[2]:
        return -1
    else:
        return 0

def SiOH_final_Sort(entry1,entry2):
    item1=entry1[3]
    item2=entry2[3]
    
    #check number of Si first
    if item1[0] > item2[0]:
        return 1
    elif item1[0] < item2[0]:
        return -1
    else:
    #then check number of O
        if item1[1] > item2[1]:
            return 1
        elif item1[1] < item2[1]:
            return -1
        else:
    #then check number of H
            if item1[2] > item2[2]:
                return 1
            elif item1[2] < item2[2]:
                return -1
            else:
                return 0

def calc_local_structure(basedf,pair_path):
    df=basedf.copy()    
    csvlist=df["csvname"].unique()

    for csvfile in csvlist:

        print("calc_local_structure on "+ str(csvfile))
        
        dfc=df[df["csvname"]==csvfile]
        #print(dfc.to_string())
        (ratio,Hnum)=stats_from_csv_name(csvfile)
        
        (posdf,box)=load_data_and_bonds_from_csv(csvfile)
        
        si_o_bondorder=0.9
        si_si_bondorder=3
        si_h_bondorder=1.7
        
        final_configs=[]
        for index, row in dfc.iterrows():
            f_loc=row['fPos']
            mover=int(row['id'].split('-')[0])
            zapped=int(row['id'].split('-')[1])
            cell_origin=row['iPos']
            initial_pair=[]
            final_pair=[]
            
            
            
            blist_i=posdf.at[mover,'bonds']
            for b in blist_i:
                bi=b[0]
                btype=posdf.at[bi,'type']
                if btype=="Si":
                    initial_pair.append(bi)
            
            blist_f=posdf.at[zapped,'bonds']
            for b in blist_f:
                bi=b[0]
                btype=posdf.at[bi,'type']
                if btype=="Si":
                    final_pair.append(bi)
                        
            mid_si=find_movers_neighbor(posdf,mover,zapped)
            
            initial_si=None
            for ip in initial_pair:
                if mid_si != ip:
                    initial_si=ip
                elif mid_si==ip:
                    fine=0
                else:
                    print('BAD calc_local_structure')
            final_si=None
            for fp in final_pair:
                if mid_si != fp:
                    final_si=fp
                elif mid_si==fp:
                    fine=0
                else:
                    print('BAD calc_local_structure')    
                
            #final_pair=find_final_si_pair(posdf,box,mover,f_loc=f_loc)
            si_neis=[initial_si,mid_si,final_si]
            # print(si_neis)
            all_atoms=[]
            
            bad_o=[mover,zapped]
            bad_si=si_neis
            bad_h=[]
            all_atoms=bad_si + bad_o
            all_neis=[]
            for si in si_neis:
                si_count=0
                o_count=0
                h_count=0
                blist_i=posdf.at[si,'bonds']
                cur_atoms=[]
                #print(blist_i)
                for b in blist_i:
                    bi=b[0]
                    cur_atoms.append(bi)
                    btype=posdf.at[bi,'type']
                    if btype=="Si" and bi not in all_atoms:# and b[1]<si_si_bondorder:
                        si_count+=1
                        all_atoms.append(bi)
                    elif btype=="O" and bi not in all_atoms:# and b[1]<si_o_bondorder:
                        o_count+=1
                        all_atoms.append(bi)
                    elif btype=="H" and bi not in all_atoms:#and b[1]<si_h_bondorder:
                        h_count+=1
                        all_atoms.append(bi)
                
                all_neis.append([si_count,o_count,h_count])
                    
                    

                
            all_atoms += bad_o + bad_si + bad_h
            feb=round(row['FEB'],3)
            reb=round(row['REB'],3)
        #Geometry:{final_bonds[0]}, {final_bonds[1]} 
    
            tmp=[mover,zapped,feb,reb,all_neis[0],all_neis[1],all_neis[2]]
            cur_fc=tmp #np.array(tmp,dtype=object)
            

            current_folder=image_folder+csvfile.removesuffix(".csv")+'/'
            if not os.path.exists(current_folder):
                os.mkdir(current_folder)
            image_name=current_folder+f"{cur_fc[0]}-{cur_fc[1]}.gif"
            mv=f"{mover}-{zapped}:"
            
            txt_overlay=f"i:    {all_neis[0][0]}Si-{all_neis[0][1]}O-{all_neis[0][2]}H\nm:  {all_neis[1][0]}Si-{all_neis[1][1]}O-{all_neis[1][2]}H\nf:    {all_neis[2][0]}Si-{all_neis[2][1]}O-{all_neis[2][2]}H\nFEB={feb} REB={reb}"
            
            plot_small_chunk(csvfile,image_name,all_atoms,[mover,zapped],cell_origin,txt_overlay,pair_path)
            
            
            final_configs.append(cur_fc)
            #print()
        from functools import cmp_to_key
        
            
        final_configs.sort(key=cmp_to_key(feb_final_sort))
        print("FEB > REB")
        for fc in final_configs:
            if fc[0] > fc[1]:
                print(f"i={fc[4][0]}Si-{fc[4][1]}O-{fc[4][2]}H   m={fc[5][0]}Si-{fc[5][1]}O-{fc[5][2]}H   f={fc[6][0]}Si-{fc[6][1]}O-{fc[6][2]}H     FEB={fc[2]} REB={fc[3]}      {fc[0]}-{fc[1]} ")
        
        print("FEB < REB")
        for fc in final_configs:
            if fc[0] < fc[1]:
                print(f"i={fc[4][0]}Si-{fc[4][1]}O-{fc[4][2]}H   m={fc[5][0]}Si-{fc[5][1]}O-{fc[5][2]}H   f={fc[6][0]}Si-{fc[6][1]}O-{fc[6][2]}H     FEB={fc[2]} REB={fc[3]}      {fc[0]}-{fc[1]} ")


from ovito.pipeline import ModifierInterface
from ovito.data import DataCollection
from ovito.modifiers import AffineTransformationModifier
from ovito.pipeline import ModifierInterface
from traits.api import Range, observe
import numpy as np
class TurntableAnimation(ModifierInterface):
    
    # Parameter controlling the animation length (value can be changed by the user):
    duration = Range(low=1, value=12)

    def compute_trajectory_length(self, **kwargs):
        return self.duration

    def modify(self, data: DataCollection, *, frame: int, **kwargs):
        
        a = data.cell[:,0]
        b = data.cell[:,1]
        c = data.cell[:,2]
        o = data.cell[:,3]
        #print(f"{a} {b} {c}")
        # Apply a rotational transformation to the whole dataset with a time-dependent angle of rotation:
        theta = np.deg2rad(frame * 360 / self.duration)
        x=o[0]+a[0]/2
        y=o[1]+b[1]/2
        z=o[2]+c[2]/2
        cost=np.cos(theta)
        sint=np.sin(theta)
        tm = [[cost, -sint, 0, -x*cost+y*sint+x],
                [sint,  cost, 0, -x*sint-y*cost+y],
                [ 0, 0, 1, z]]
        data.apply(AffineTransformationModifier(transformation=tm))

    # This is needed to notify the pipeline system whenever the animation length is changed by the user:
    @observe("duration")
    def anim_duration_changed(self, event):
        self.notify_trajectory_length_changed()
        
        
class ShrinkWrap(ModifierInterface):
    
    def modify(self, data: DataCollection, *, frame: int, **kwargs):  
            # There's nothing we can do if there are no input particles.
            if not data.particles or data.particles.count == 0:
                return

            # Compute min/max range of particle coordinates.
            coords_min = np.amin(data.particles.positions, axis=0)
            coords_max = np.amax(data.particles.positions, axis=0)
            

            # Build the new 3x4 cell matrix:
            #   (x_max-x_min  0            0            x_min)
            #   (0            y_max-y_min  0            y_min)
            #   (0            0            z_max-z_min  z_min)
            matrix = np.empty((3,4))
            matrix[:,:3] = np.diag(coords_max - coords_min)
            matrix[:, 3] = coords_min

            # Assign the cell matrix - or create whole new SimulationCell object in
            # the DataCollection if there isn't one already.
            data.create_cell(matrix, (False, False, False))
    
    
    
# #Plotters
def plot_small_chunk(csv_file,imagename,all_atoms,select_atoms,cell_origin,text_overlay,pair_path):
    from ovito.io import import_file, export_file
    import ovito.data
    import ovito.modifiers
    from ovito import scene
    from ovito.vis import TextLabelOverlay, Viewport, PythonViewportOverlay, CoordinateTripodOverlay
    from ovito.qt_compat import QtCore 
    from ovito.vis import TachyonRenderer
    from ovito.qt_compat import QtGui
    from PySide6.QtGui import QFont, QFontDatabase, QFontInfo


    
    #clear the scene in case it isnt already
    for pipe in scene.pipelines:
        pipe.remove_from_scene()
        
    dfile=pair_path+csv_file.removesuffix(".csv")+".dat"

    if not os.path.exists(dfile):
        print(f'plot_small_chunk fail because {dfile} does not exist')
                
    pipeline=import_file(dfile)
    
    expr=""
    for atom in all_atoms:
        expr+=f"ParticleIdentifier=={atom}||"
    expr=expr.removesuffix("||")

    pipeline.modifiers.append(ovito.modifiers.ExpressionSelectionModifier(expression = expr))
    pipeline.modifiers.append(ovito.modifiers.InvertSelectionModifier())
    pipeline.modifiers.append(ovito.modifiers.DeleteSelectedModifier())
    pipeline.modifiers.append(ovito.modifiers.ClearSelectionModifier())
    
    expr=""
    for atom in select_atoms:
        expr+=f"ParticleIdentifier=={atom}||"
    expr=expr.removesuffix("||")
    pipeline.modifiers.append(ovito.modifiers.ExpressionSelectionModifier(expression = expr))
    pipeline.modifiers.append(ovito.modifiers.AssignColorModifier(color=(0, 1, 0)))
    # #pipeline.modifiers.append(ovito.modifiers.AffineTransformationModifier(operate_on = {'cell'}, # Transform particles but not the box.
    #                                                                         transformation = [[1, 0, 0, cell_origin[0]],
    #                                                                                             [0, 1, 0, cell_origin[1]],
    #                                                                                             [0, 0, 1, cell_origin[2]]]))
    pipeline.modifiers.append(ShrinkWrap())
    pipeline.modifiers.append(TurntableAnimation())
    
    

    
    

    # Create the overlay:
    print(text_overlay)
    

    pipeline.add_to_scene()
    data=pipeline.compute()

    data.cell.vis.enabled = False  

    vp = Viewport(type=Viewport.Type.Front)
    
    
    # overlay = TextLabelOverlay(text=text_overlay,font_size=0.05,text_color=(1,0,0), alignment=QtCore.Qt.AlignTop ^ QtCore.Qt.AlignLeft)
    # vp.overlays.append(overlay)

    imagesize=(600,600)
    vp.zoom_all(size=imagesize)


    #image=vp.render_image(filename=imagename,size=imagesize,renderer=TachyonRenderer(ambient_occlusion=False, shadows=False))
    image=vp.render_anim(filename=imagename,fps=1,size=imagesize,renderer=TachyonRenderer(ambient_occlusion=False, shadows=False))
    pipeline.remove_from_scene()
    
    
    from PIL import Image, ImageFont, ImageDraw, ImageSequence
    import io
    im=Image.open(imagename)


    fnt=ImageFont.truetype("Arial", 32)
    color=(0,0,0,255)
    frames=[]
    duration=[]
    for i in range(im.n_frames):
        im.seek(i)
        frame = im.convert('RGBA').copy()
        duration.append(im.info['duration'])
        # Draw the text on the frame
        d = ImageDraw.Draw(frame)
        d.text((0,0),text_overlay,font=fnt,fill=color)
        del d

        # However, 'frame' is still the animated image with many frames
        # It has simply been seeked to a later frame
        # For our list of frames, we only want the current frame

        # Saving the image without 'save_all' will turn it into a single frame image, and we can then re-open it
        # To be efficient, we will save it to a stream, rather than to file
        b = io.BytesIO()
        frame.save(b, format="GIF")
        frame = Image.open(b)

        # Then append the single frame image to a list of frames
        frames.append(frame)
    # Save the frames as a new image
    frames[0].save(imagename, save_all=True, append_images=frames[1:])

    
    

def plot_any_split_hist(basedf,col,var,colranges,numbins,vartitle,xlabel,units,subfolder,plotfilename):
    df=basedf.copy()
    csvlist=df["csvname"].unique()
    
    numregion=len(colranges)-1
    ylabel=f"Number of {col} in range"
    
    avgstdlist=[]
    for k in range(numregion):
            avgstdlist.append([])
            
    t=[]
    for j in range(len(csvlist)):
        csvfile=csvlist[j]
        (ratio,Hnumber)=stats_from_csv_name(csvfile)
        t.append([float(ratio),int(Hnumber),csvfile])
    tmp=np.array(t,dtype=object)
    
    sortedcsv=tmp[np.lexsort((tmp[:,1],tmp[:,0]))]

    
    
    uniratio=np.unique(sortedcsv[:,0])

    
    for ratio in uniratio:
        ratiocsvs=sortedcsv[sortedcsv[:,0]==ratio]
        
        numcsv=len(ratiocsvs)

        figure=plt.figure(constrained_layout=True,figsize=[numregion*3,numcsv*3])
        figs = figure.subfigures(ncols=1,nrows=numcsv)
        
        for j in range(numcsv):
            csvfile=ratiocsvs[j][2]
            (ratio,Hnumber)=stats_from_csv_name(csvfile)

            dfc=df[df["csvname"]==csvfile]
            fig=figs[j] if numcsv > 1 else figs
            
            axisfontsize=6
            fig.suptitle(f"{col}({len(dfc)})-{vartitle} histogram, x={ratio} and {Hnumber} H",fontsize=axisfontsize+2)
            axes=fig.subplots(ncols=numregion,nrows=1)

            minval=dfc[var].min()
            maxval=dfc[var].max()
            
            for i in range(numregion):
                ax=axes[i]
                alow=colranges[i]
                ahigh=colranges[i+1]
                alist=dfc.loc[dfc[col].between(alow,ahigh),var]
                
                numfeb=len(alist)
                avga=np.mean(alist)
                stdd=np.std(alist)
                title=f"{numfeb} {col}'s between {alow} eV and {ahigh} eV, avg: {round(avga,1)}{units}"
                
                counts, bins = np.histogram(alist,bins=numbins,range=(minval,maxval))
                
                ax.hist(bins[:-1], bins, weights=counts)
                
                ax.tick_params(axis='both', which='major', labelsize=axisfontsize)
                ax.tick_params(axis='both', which='minor', labelsize=axisfontsize)
                ax.set_xlabel(xlabel,fontsize=axisfontsize)
                ax.set_ylabel(ylabel,fontsize=axisfontsize)
                ax.set_title(title,fontsize=axisfontsize+1)
                
                avgstdlist[i].append([float(ratio),int(Hnumber),avga,stdd])
                i+=1
                    
                
        name=csvfile.split('/')[-1].removesuffix('.csv')
        dirname=image_folder+subfolder+'/'+col+'/'
        os.makedirs(dirname, exist_ok=True)
        figname=dirname+plotfilename+'-'+ratio+'.svg'
        
        figure.savefig(figname)
        print(f"Saved figure with path {figname}")

        
        plt.close(figure)

    
    avglist=np.array(avgstdlist)

    
    
    fig, axes = plt.subplots(ncols=numregion,nrows=1, figsize=[numregion*5,5])  
    supfigname=dirname+plotfilename+'.svg'
    minv=np.min(avglist[:,:,2])
    maxv=np.max(avglist[:,:,2])
    limbuf=(maxv-minv)*.1
    for i in range(numregion):
        avgs=avglist[i]

        ax=axes[i]
        #al=avgstdlist[i]
        alow=colranges[i]
        ahigh=colranges[i+1]
        
        for ur in uniratio:
            #print(f"{type(a[0])} {type(a[1])} {type(a[2])} {type(a[3])}")
            a=avgs[avgs[:,0]==ur]

            ax.scatter(HNumToConcentration*a[:,1],a[:,2],label=str(ur))
        
        # get legend handles and their corresponding labels
        handles, labels = ax.get_legend_handles_labels()

        # zip labels as keys and handles as values into a dictionary, ...
        # so only unique labels would be stored 
        dict_of_labels = dict(zip(labels, handles))

        # use unique labels (dict_of_labels.keys()) to generate your legend
        ax.legend(dict_of_labels.values(), dict_of_labels.keys(),title="SiOx Ratio")
        
        
        ax.set_xscale('log')
        
        ax.set_ylim(minv-limbuf,maxv+limbuf)
        ax.set_title(f"{vartitle} between {alow} eV and {ahigh} eV")
        ax.set_xlabel('Areal Hydrogen Density(H/cm$^2$)')
        ax.set_ylabel(f"Avg {vartitle}")
        
    fig.savefig(supfigname)
    print(f"Saved figure with path {supfigname}")
    plt.show()
    
    plt.close(fig)
    
    
def plot_pair_angle(basedf,col="FEB"):
    df=basedf.copy()
    csvlist=df["csvname"].unique()
    subfolder="angle-OSiO-bond/"
    colranges=[0,1.8,2.8,3.8,10]
    valcol="Pair angle"
    print('here')
    df=angle_between_pts_df(df,valcol)
    #print(df.to_string())
    
    units="°" 
    xlabel= f"Bond angle{units} of O pairs" 
    
    plot_multi_distribution(df,valcol, units)
    numbins=10
    
    vartitle=f"angle{units} between pairs"
    plotfilename=f"bondangle-histogram"
    
    print("Calculation done, now plotting")
    t=[]
    for j in range(len(csvlist)):
        csvfile=csvlist[j]
        (ratio,Hnumber)=stats_from_csv_name(csvfile)
        t.append([float(ratio),int(Hnumber),csvfile])
    tmp=np.array(t,dtype=object)
    
    sortedcsv=tmp[np.lexsort((tmp[:,1],tmp[:,0]))]

    
    
    uniratio=np.unique(sortedcsv[:,0])

    numregions=len(colranges)-1
    fig,axes = plt.subplots(ncols=1,nrows=1,figsize=(6,6))
    
    #dfl=df[df[valcol]>80]
    
    for ratio in uniratio:
        ax1=axes
        ratiocsvs=sortedcsv[sortedcsv[:,0]==ratio]
        
        numcsv=len(ratiocsvs)
        al=[]
        fl=[]
        for j in range(numcsv):
            csvfile=ratiocsvs[j][2]
            dfr=df[df["csvname"]==csvfile]

            (r,Hnumber)=stats_from_csv_name(csvfile)
            
            # alist=dfr.loc[dfr[col].between(alow,ahigh),valcol].tolist()
            # print(alist)
            # flist=alist=dfr.loc[dfr[col].between(alow,ahigh),col].tolist()
            alist=dfr[valcol].tolist()
            flist=dfr[col].tolist()
            
            al.extend(alist)
            fl.extend(flist)
            
            #rlist=dfr[]
            numfeb=len(alist)
            title=f"Migration pair angles v. {col} for SiOx with x={ratio} and {Hnumber}H." # - between {alow} eV and {ahigh} eV"
        
        
        ax1.scatter(al,fl,s=6,label=ratio)#,facecolors='none',linewidths=0.4)
        ax1.set_title(title)
        ax1.set_xlabel(f"O-Si-O pair angle({units})")
        ax1.set_ylabel(f"{col}(eV)")
        # get legend handles and their corresponding labels
        
        
    avgSiO2=109.4
    plt.axvline(avgSiO2,linestyle="dashed")
    plt.text(110,.9,f'Average O-Si-O bond angle in a-SiO2:{avgSiO2}{units}', size=9)
    handles, labels = ax1.get_legend_handles_labels()

    # zip labels as keys and handles as values into a dictionary, ...
    # so only unique labels would be stored 
    dict_of_labels = dict(zip(labels, handles))

    # use unique labels (dict_of_labels.keys()) to generate your legend
    ax1.legend(dict_of_labels.values(), dict_of_labels.keys())
          
    plt.show()
    return df
    
def plot_multi_distribution(setdf,cols=["FEB","REB"],units="eV",plot=False):
    df=setdf.copy()
    colstxt=""
    
    numcol=len(cols)
    fig, axes=plt.subplots(2,numcol,sharex=True,height_ratios=[15,1])
    
    
    if numcol>1:
        for i in range(numcol):
            if i!=0:
                a1=axes[0,0]
                a1.get_shared_y_axes().join(a1, axes[0,i])
            
            colstxt+=cols[i]
            if i == numcol-2:
                colstxt+=" and "
            elif i == numcol-1:
                colstxt+=""
            else:
                colstxt+=", "
    else:
        colstxt=cols[0]

            
    fig.subplots_adjust(hspace=0.1) 
        
    #fig, axes=plt.subplots(2,numcol,height_ratios=[15,1])
    for i in range(numcol):  
        col=cols[i]
        tdf=dist_from_df(df,col,units,plot)

        if numcol>1:
            ax1=axes[0,i]
            ax2=axes[1,i]
        else:
            ax1=axes[0]
            ax2=axes[1]
        #1%, 3%, 5%, 6%, 7%, 9%, 11%, 13%
        
        
        t=0


        ratios=tdf[f"{col}ratio"].unique()
        
        for r in ratios:


            dfr=tdf[tdf[f"{col}ratio"]==r]

            dfr=dfr.sort_values(by=[f"{col}Hnum"])

            xvals=[]#np.array([0,1,3,5,6,7,9,11,13])

            min=10
            max=0
            
            xvals=dfr[f"{col}Hnum"].to_numpy(dtype=float)
            yvals=dfr[f"{col}mean"].to_numpy(dtype=float)
            xvals=xvals*HNumToConcentration
            ax1.set_xscale('log')
            ax2.set_xscale('log')
            #Setting tick and label text to smaller font
            axisfontsize=10
            

    
            ax2.tick_params(axis='both', which='major', labelsize=axisfontsize)
            ax2.tick_params(axis='both', which='minor', labelsize=axisfontsize)
            ax1.tick_params(axis='both', which='major', labelsize=axisfontsize)
            ax1.tick_params(axis='both', which='minor', labelsize=axisfontsize)
            
            

            fig.supxlabel('Areal Hydrogen Density(H/cm$^2$)',fontsize=axisfontsize)
            ax1.set_title(f"{col}",fontsize=axisfontsize)
            ax1.set_ylabel(f'{col} - mean ({units})',fontsize=axisfontsize)

            ax2.set_yticks((0,10))
            ax2.set_ylim(0,0.5)
            # ax1.set_ylim(min-.1,max+.1)

            if i!=0:
                ax1.tick_params(labelleft=False)

            ax1.spines.bottom.set_visible(False)
            ax2.spines.top.set_visible(False)
            ax1.xaxis.tick_top()
            



            ax1.tick_params(labeltop=False)  # don't put tick labels at the top
            ax2.xaxis.tick_bottom()


            d = .5  # proportion of vertical to horizontal extent of the slanted line
            kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                        linestyle="none", color='k', mec='k', mew=1, clip_on=False)
            ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
            ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

            #make sure that 0 shows up fine on the log plot
            if xvals[0]==0:
                low=xvals[1]
                low=low/2
                xvals[0]=low
                ax1.set_xlim(low,xvals[-1]*1.1)


            erb=ax1.scatter(xvals, yvals, label=str(r))#,yerr=yer[1])
            ax1.legend(title="Value of x")
            t+=1
        
        
        
        # avgSiO2=109.4
        # ax1.axhline(avgSiO2,linestyle="dashed")
        # ax1.text(0.3,.08,f'Average O-Si-O bond angle in a-SiO2:{avgSiO2}{units}', size=9,transform=ax1.transAxes)
    fig.suptitle(f'Mean {colstxt} for Oxygen migration in SiOx',fontsize=axisfontsize+1)
        
    name=f"{col}vH"
    dirname=image_folder+"/individual-dist/"
    os.makedirs(dirname, exist_ok=True)
    figname=dirname+name+'.svg'
    print(f"Saved figure with path {figname}")
    fig.savefig(figname)
    plt.close(fig)
        #erb[-1][0].set_linestyle('--')
     

def plot_rangehist(basedf,rlist,type=None,col='FEB',skip=False): 
    df=basedf.copy()    
    csvlist=df["csvname"].unique()
    
    colranges=[0,1.8,2.8,3.8,10]
    if type is None:
        typename="Atoms"
    else:
        typename=type
        
    subfolder=f"atomcount-split/{typename}/"
    j=0
    for csvfile in csvlist:
        # if j==0:
        #     j+=1
        #     continue
        print("plot_rangehist on "+ str(csvfile))
        filetimestart = time.time()
        
        dfc=df[df["csvname"]==csvfile]
        #print(dfc.to_string())
        (ratio,Hnum)=stats_from_csv_name(csvfile)
        
        #(posdf,box)=load_data_and_bonds_from_csv(csvfile)
        (posdf,box)=load_data_from_csv(csvfile)
        if type is not None:
            posdf=posdf[posdf["type"]==type]
        
        feblist=[]#dfc[col].to_numpy()
        i=0
        
        if skip:
            skipping=True
            for radius in rlist:
                valcol=f'{typename}range{radius}'
                if valcol not in dfc.columns:
                    skipping=False
                else:
                    # print(dfc[valcol])
                    # print(dfc[valcol].isnull().values.any())
                    skipping=not (dfc[valcol].isnull().values.any())
            if skipping:
                print(f"Skipping {csvfile}")
                continue
        
        for index, row in dfc.iterrows():
            rowtimestart=time.time()
            curcol=row[col]
            pair=row['pair']
            pos=row['iPos']
            mover=int(pair.split('-')[0])
            
            
# ##Testing
#             if row[col] < colranges[1] or row[col]>colranges[2]:
#                 continue
# ##Testing
            #need to load in this csv's data file and look at that dataframe in conjunction with this one
            distdf=apply_dist_from_pos(posdf,box,pos,type)
            
            for radius in rlist:
                valcol=f'{typename}range{radius}' 
                df.at[index,valcol]=int(len(distdf[distdf['dist']<radius]))
                
            rowtimeend=time.time()
            if i == 0 and me == 0:
                rowtime=rowtimeend-rowtimestart
                esttime=rowtime*len(dfc)
                if esttime > 10:
                    print(f"EST total time for this csv: {esttime}")
                i+=1
        
    units=r"$\AA$"
    print("Calculation done, now plotting")
    for radius in rlist:
        valcol=f'{typename}range{radius}' 
        
        xlabel= f"Number of {typename} atoms within {radius}{units}" if typename=="Atoms" else f"Number of atoms within {radius}{units}"
        dfr = df[df['dist']<radius]
        numbins=10
        
        vartitle=f"{typename} within {radius}{units}"
        plotfilename=f"{typename}-{valcol}-histogram"
        
        plot_any_split_hist(dfr,col,valcol,colranges,numbins,vartitle,xlabel,units,subfolder,plotfilename)
            

    return df 
    
#look at all pairs and find how many atoms(of a certain type) are within a range
def plot_atominrange(basedf,radius,type=None,col='FEB'): 
    df=basedf.copy()    
    csvlist=df["csvname"].unique()
    subfolder="atomcount/"

    
    for csvfile in csvlist:
        filetimestart = time.time()
        
        dfc=df[df["csvname"]==csvfile]
        (ratio,Hnum)=stats_from_csv_name(csvfile)
        
        (posdf,box)=load_datafile_from_csv(csvfile)
        
        if type is not None:
            posdf=posdf[posdf["type"]==type]
        
        
        feblist=[]#dfc[col].to_numpy()
        i=0
        for index, row in dfc.iterrows():
            rowtimestart=time.time()
            curcol=row[col]
            pair=row['pair']
            pos=row['iPos']
            mover=int(pair.split('-')[0])

            
            #need to load in this csv's data file and look at that dataframe in conjunction with this one
            dfc.at[index,'inrange']=atoms_in_radius(posdf,box,pos,radius,type)
            rowtimeend=time.time()
            if i == 0 and me == 0:
                rowtime=rowtimeend-rowtimestart
                esttime=rowtime*len(dfc)
                if esttime > 10:
                    print(f"EST total time for this csv: {esttime}")
                i+=1
            

        
    
        # types=dfc['type'].unique()
        # colormap=cm.virdis
        # colorlist = [colors.rgb2hex(colormap(i)) for i in np.linspace(0, 0.9, len(types))]
        
        # for i,c in enumerate(colorlist):
        avgfebdf=dfc.groupby(["inrange"],axis=0)[col].mean().reset_index()
        feblist=dfc[col].to_numpy()
        numlist=dfc['inrange'].to_numpy()
        
        if type is None:
            type="Atoms"
            
            
        if  me==0:
            radstr=f"{radius}ang"
            title=f"{col} vs. #{type} within {radstr} for x={ratio} and {Hnum} Hydrogens"
            fig,ax=plt.subplots(1,1)
            s=15
            ax.scatter(numlist,feblist,s=s,facecolors='none',edgecolors='b',linewidths=0.4)
            for i,r in avgfebdf.iterrows():
                ax.scatter(r[0],r[1],marker='x',color='r',s=s)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            ax.set_xlabel(f"Number of {type} within {radius} angstroms")
            ax.set_ylabel(f"{col}(eV)")
            ax.set_title(title)
            name=csvfile.split('/')[-1].removesuffix('.csv')
            dirname=image_folder+subfolder+radstr+'/'+type+'/'
            os.makedirs(dirname, exist_ok=True)
            figname=dirname+name+'.svg'
            print(f"Saved figure with path {figname}")
            fig.savefig(figname)
            plt.close(fig)
            
            filetimeend=time.time()
            runtime=filetimeend-filetimestart
            dflen=len(dfc)
            print(f"{name} took {runtime}s, avg run {runtime/dflen}s")
            print('-----------------------------------------------------')
    return dfc


def plot_vang_multi(basedf,col='FEB',refvec=[0,0,1]):
    df=basedf.copy()
    #df=df.sort_values(by='csvname')
 
    
    subfolder="angle-relative-interface/"
    
    var="ang"

    
    colranges=[0,1.8,2.8,3.8,10]
    
    dfa=angle_between_df(df,var,refvec)
    dfa[var]=abs(dfa[var].apply(lambda x: 90 - x))
    # for j in range(numcsv):
    #     csvfile=csvlist[j]
    #     dfc=df[df["csvname"]==csvfile]

    #     (ratio,Hnumber)=stats_from_csv_name(csvfile)
        
    xlabel=f"Angle(°) relative to {str(refvec)} plane."
    
    numbins=10
    units="(°)"
    vartitle="Angle(°) to i-face"
    plotfilename="angle-histogram"
    print("Calculation done, now plotting")
    plot_any_split_hist(dfa,col,var,colranges,numbins,vartitle,xlabel,units,subfolder,plotfilename)
 
def plot_bondang_vdz(basedf):
    df=basedf.copy()
    csvlist=df["csvname"].unique()
    subfolder="angle-OSiO-bond/"
    colranges=[0,1.8,2.8,3.8,10]
    valcol="Pair angle"
    #print(df.to_string())
    #box=np.array(df.iloc[0]['box'])
    
    hnum=[]
    angnum=[]
    meanlist=[]
    for csvfile in csvlist:
        dfc=df[df["csvname"]==csvfile]
        #print(dfc.to_string())
        (ratio,Hnum)=stats_from_csv_name(csvfile)
        

        #dfc=dfc[dfc.apply(limit_zpos_iface,axis=1,args=(bulk,))]
        
        (posdf,box)=load_data_from_csv(csvfile)
        
        dfc=angle_between_pts_df(dfc,valcol)
        dfc=dfc.apply(vz_idz,axis=1)
        #dfc=dfc[dfc[valcol]<maxang]
        
        
            
    zpos=dfc['zpos'].tolist()
    anglis=dfc['Pair angle'].tolist()    
    fig, ax = plt.subplots()
    # pl=np.array([hnum,angnum]).T
    ax.scatter(zpos,anglis,s=6,facecolors='none',edgecolors='b')
    # ax.set_title(f"Number of H in range vs. Number of pairs with angle < {maxang}")
    # ax.set_xlabel(f"Number of H within {dist}Å of pair.")
    # ax.set_ylabel(f"Number of pair angles < 110° in range.")
    # for hn in np.unique(pl[:,0]):
    #     tl = pl[pl[:,0]==hn]
    #     mtl=np.mean(tl[:,1])
    #     ax.scatter(hn,mtl,s=20,marker='x',c='r')
    
def create_bond_angles(atoms,box,type1='O',typeM='Si',type2='O'):
    #create bond angles for a type1-typeM-type2 bond ex: O-Si-O
    #print(atoms.to_string())
    t1atoms=atoms[atoms['type']==type1]
    bangleList=[]
    
    for i, row in t1atoms.iterrows():
        nbonds = row['bonds']

        
        tmn=[]

        #create the list of type 2
        for n in nbonds:
            ni=n[0]
            bo=n[1]
            
            if ni not in atoms.index:
                continue
            neitype=atoms.at[ni,'type']

            if neitype == typeM:
                if bo < SiOBondOrder:
                    continue
                tmn.append(ni)
        
        #run through the bonds of the middle atom and fill a list with any type 3 neighbors 
        for tmi in tmn:
            neibonds = atoms.at[tmi,'bonds']
            
            for neib in neibonds:
                nei=neib[0]
                neibo=neib[1]
                
                if nei not in atoms.index:
                    continue
                
                if neibo < SiOBondOrder:
                    continue

                if nei ==i:
                    continue
                
                neitype = atoms.at[nei,'type']
                if neitype != type2:
                    continue

                
  
                p1=[i, tmi, nei]
                p2=[nei, tmi, i]

                #good pair if it got this far
                #add this pair to the pair list              
                if p1 not in bangleList and p2 not in bangleList:
                    bangleList.append(p1)
                    
    #now we have a list of all acceptable bonds
    retlist=[]
    for trio in bangleList:
        i1=trio[0]
        im=trio[1]
        i2=trio[2]
        
        pos1=atoms.at[i1,'pos']
        posm=atoms.at[im,'pos']
        pos2=atoms.at[i2,'pos']
        (ang,[v1,v2]) = angle_between_pts(box,pos1,pos2,posm)
        retlist.append((posm,ang[0]))

    return retlist
        

def plot_all_bondang_vs_atom(basedf,dist,atype="H",maxang=180,all=False,bulk=False):
    #not just doing bond angle on our pairs but every atom in a sim
    df=basedf.copy()
    csvlist=df["csvname"].unique()
    subfolder="angle-all-OSiO-bond/"
    colranges=[0,1.8,2.8,3.8,10]
    valcol="Pair angle"
    #print(df.to_string())
    #box=np.array(df.iloc[0]['box'])
    
    hnum=[]
    angnum=[]
    anglist=[]
    for csvfile in csvlist:
        dfc=df[df["csvname"]==csvfile]
        
        print(csvfile)
        (ratio,Hnum)=stats_from_csv_name(csvfile)
        
        

        
        
        
        (posdf,box)=load_data_and_bonds_from_csv(csvfile)
        if not all:
            posdf=posdf[posdf.apply(limit_zpos_iface,axis=1,args=(bulk,'pos',))]
        
        bondAngleL=create_bond_angles(posdf,box)
        print(len(bondAngleL))
        
        #dfc=angle_between_pts_df(dfc,valcol)
        #dfc=dfc.apply(vz_idz,axis=1)
        #dfc=dfc[dfc[valcol]<maxang]
        
        for ba in bondAngleL:
            
            pos=ba[0]
            an=ba[1]
            # if an > maxang:
            #     continue
            #print("plot_bondang_vdz on "+ str(csvfile)
            
            #fulldf=apply_dist_from_pos(dfc,box,pos,col='iPos')
            hdist=apply_dist_from_pos(posdf,box,pos,atype)
            
            numH=0
            for ind, row in hdist.iterrows():
                if row['type']==atype and row['dist']<dist:
                    numH+=1
            
            
            #angdf=fulldf[fulldf[valcol]<maxang]
            #meanpa=np.mean(pa)
            hnum.append(numH)
            #angnum.append(len(angdf))
            anglist.append(an)
            
            # plt.hist(bins[:-1],bins,weights=counts)
            # plt.title(f"{len(fulldf)} Pair angles around {str(pos)}, avg:{round(meanpa,2)} - {numH}H nearby")
            # plt.show()
            
    if maxang != 180:
        fig, ax = plt.subplots()
        pl=np.array([hnum,anglist]).T
        # ax.scatter(hnum,angnum,s=6,facecolors='none',edgecolors='b')
        title=f"# of {atype} in range vs. % of pairs with angle < {maxang}° "
        
        if not all:
            title+="in bulk " if bulk else  "near interface "
        #title+=f"for x={ratio} w/ {Hnum}H"
        ax.set_title(title)
        ax.set_xlabel(f"Number of {atype} within {dist}Å.")
        ax.set_ylabel(f"% of pair angles in range < 110°")
        mplth=[]
        mpltCount=[]
        
        #run through all the unique "number of H in range" values
        for hn in np.unique(pl[:,0]):
            tl = pl[pl[:,0]==hn]#list of angles with this number of H in range
            tot=0
            cnt=0
            for a in tl[:,1]:
                tot+=1
                if a < maxang:
                    cnt+=1
                    
            mtl=cnt/tot
            mplth.append(hn)
            mpltCount.append(mtl)
            
        print(mplth)
        print(mpltCount)
        ax.plot(mplth,mpltCount)
    
    
    fig1, ax1 = plt.subplots()
    
    title=f"# of {atype} in range vs. O-Si-O bond angle "
    if not all:
        title+="in bulk " if bulk else  "near interface "
    #title+=f"for x={ratio} w/ {Hnum}H"
    ax1.set_title(title)
    ax1.set_xlabel(f"# of {atype} within {dist}Å of bond")
    ax1.set_ylabel(f"O-Si-O bond angle(°)")
    ax1.scatter(hnum,anglist,s=6,facecolors='none',edgecolors='b')
    pl=np.array([hnum,anglist]).T
    mplth=[]
    mpltAvg=[]
    for hn in np.unique(pl[:,0]):
        tl = pl[pl[:,0]==hn]
        mtl=np.mean(tl[:,1])
        mplth.append(hn)
        mpltAvg.append(mtl)
        print(f"{hn} H in range - avg bond angle: {mtl}")
        
    ax1.plot(mplth,mpltAvg,c='r')#s=20,marker='x',
    plt.show()

def plot_bondang_vs_h(basedf,poslist,dist,maxang=180,bulk=False):
    df=basedf.copy()
    csvlist=df["csvname"].unique()
    subfolder="angle-OSiO-bond/"
    colranges=[0,1.8,2.8,3.8,10]
    valcol="Pair angle"
    #print(df.to_string())
    #box=np.array(df.iloc[0]['box'])
    
    hnum=[]
    angnum=[]
    meanlist=[]
    for csvfile in csvlist:
        dfc=df[df["csvname"]==csvfile]
        #print(dfc.to_string())
        (ratio,Hnum)=stats_from_csv_name(csvfile)
        

        dfc=dfc[dfc.apply(limit_zpos_iface,axis=1,args=(bulk,))]
        
        (posdf,box)=load_data_from_csv(csvfile)
        
        dfc=angle_between_pts_df(dfc,valcol)
        dfc=dfc.apply(vz_idz,axis=1)
        #dfc=dfc[dfc[valcol]<maxang]
        
        for pos in poslist:
            #print("plot_bondang_vdz on "+ str(csvfile)
            
            fulldf=apply_dist_from_pos(dfc,box,pos,col='iPos')
            hdist=apply_dist_from_pos(posdf,box,pos,"H")
            numH=0
            for ind, row in hdist.iterrows():
                if row['dist']<dist:
                    numH+=1
            
            fulldf=fulldf[fulldf['dist']<dist]
            
            
            # #print(fulldf.to_string())
            # zpos=fulldf['zpos'].tolist()
            pa=fulldf[valcol].tolist()
            
            #fig, ax = plt.subplots()
            # plt.scatter(zpos,pa,s=5,facecolors='none',edgecolors='b')
            #counts, bins = np.histogram(pa)
            
            angdf=fulldf[fulldf[valcol]<maxang]
            meanpa=np.mean(pa)
            hnum.append(numH)
            angnum.append(len(angdf))
            meanlist.append(meanpa)
            
            # plt.hist(bins[:-1],bins,weights=counts)
            # plt.title(f"{len(fulldf)} Pair angles around {str(pos)}, avg:{round(meanpa,2)} - {numH}H nearby")
            # plt.show()
            
            
    fig, ax = plt.subplots()
    pl=np.array([hnum,angnum]).T
    ax.scatter(hnum,angnum,s=6,facecolors='none',edgecolors='b')
    ax.set_title(f"Number of H in range vs. Number of pairs with angle < {maxang}")
    ax.set_xlabel(f"Number of H within {dist}Å of pair.")
    ax.set_ylabel(f"Number of pair angles < 110° in range.")
    for hn in np.unique(pl[:,0]):
        tl = pl[pl[:,0]==hn]
        mtl=np.mean(tl[:,1])
        ax.scatter(hn,mtl,s=20,marker='x',c='r')
    
    
    fig1, ax1 = plt.subplots()
    ax1.set_title(f"Number of H in range vs. average pair angle")
    ax1.set_xlabel(f"Number of H within {dist}Å of pair.")
    ax1.set_ylabel(f"Average pair angle(°).")
    ax1.scatter(hnum,meanlist,s=6,facecolors='none',edgecolors='b')
    pl=np.array([hnum,meanlist]).T
    for hn in np.unique(pl[:,0]):
        tl = pl[pl[:,0]==hn]
        mtl=np.mean(tl[:,1])
        ax1.scatter(hn,mtl,s=20,marker='x',c='r')
    plt.show()

def limit_zpos_iface(row,bulk,col='iPos'):
    zipos = np.array(row[col])[2]
    if zipos > 20 and zipos <28:#adjust for any positions on the z periodic boudary
        return bulk
    return not bulk


 
def vz_idz(row,var='zpos'):
    #read the position of the pair and create a new column for the z coordinate
    zipos = np.array(row['iPos'])[2]
    if zipos < 10:#adjust for any positions on the z periodic boudary
        box=np.array(row['box'],dtype=float)
        if zipos < 10:
            zipos+=box[2,1]-box[2,0]
        
    row[var]=zipos
    return row

def vz_dz(row,var='zpos'):
    zipos = np.array(row['iPos'])[2]
    zfpos = np.array(row['fPos'])[2]
    if zipos < 10 or zfpos < 10:#adjust for any positions on the z periodic boudary
        box=np.array(row['box'],dtype=float)
        if zipos < 10:
            zipos+=box[2,1]-box[2,0]
        if zfpos < 10:
            zfpos+=box[2,1]-box[2,0]
        
    row[var]=(zfpos+zipos)/2
    return row



     
def plot_vz_df(basedf,col='FEB'):
    df=basedf.copy()  
    colranges=[0,1.8,2.8,3.8,10]
    var='zpos'
    subfolder="vs-depth/"
    
    dfz=df.apply(vz_dz,axis=1)

    units=r"($\AA$)"
    xlabel=f"Z Depth({units})"
    ylabel=f"Number of {col} in range"
    numbins=20
    vartitle=r"Z Depth($\AA$)"
    plotfilename="z-depth"
    print("Calculation done, now plotting")
    plot_any_split_hist(dfz,col,var,colranges,numbins, vartitle,xlabel,units,subfolder,plotfilename)


def plot_vang_2dhist(basedf,col='FEB',refvec=[0,0,1]):
    df=basedf.copy()   
    csvlist=df["csvname"].unique()
    subfolder="angle-2d-histogram/"
    for csvfile in csvlist:
        dfc=df[df["csvname"]==csvfile]

        (ratio,Hnumber)=stats_from_csv_name(csvfile)
        
        
        
        dfa=angle_between_df(dfc,"ang",refvec)
        dfa['ang']=dfa['ang'].apply(lambda x: 90 - x)
        title=f"{col}-Angle 2D Histogram({len(dfc)}), x={ratio} and {Hnumber} H"

        
        feblist=dfc[col].to_numpy()
        anglist=abs(dfc['ang'].to_numpy())
        
        xbin=5
        ybin=11
        h,xedges,yedges=np.histogram2d(anglist,feblist,bins=(xbin,ybin))
        hist=h.T
        (xmin,xmax)=(min(xedges),max(xedges))
        (ymin,ymax)=(min(yedges),max(yedges))
        xbinwidth=(xmax-xmin)/xbin
        ybinwidth=(ymax-ymin)/ybin
        #print(f"h {hist}, x {xedges}, y {yedges}")
        fig, ax = plt.subplots(figsize=(10,6) ,subplot_kw={"projection": "3d"})
        ax.dist=13
        #f,ax = plt.subplots()
        X, Y = np.meshgrid(xedges[:-1]+xbinwidth/2,yedges[:-1]+ybinwidth/2)

        ax.plot_surface(X,Y,hist, cmap=plt.cm.Reds)
        ax.view_init(elev=60, azim=-90, roll=0)
        #ax.pcolormesh(xedges,yedges,hist,cmap='rainbow')
        # plt.scatter(anglist,feblist)
        ax.set_xlabel(f"Angle(°) relative to {str(refvec)} plane.")
        ax.set_ylabel(f"{col}(eV)")
        ax.set_title(title)
        
        name=csvfile.split('/')[-1].removesuffix('.csv')
        dirname=image_folder+subfolder+'/'+col+'/'
        os.makedirs(dirname, exist_ok=True)
        figname=dirname+name+'.svg'
        print(f"Saved figure with path {figname}")
        fig.savefig(figname)
        plt.close(fig)
 

def dist_from_csv(csvfile, plot=False):
    data = csv_to_df(csvfile)
    dist_from_df(data,plot=plot)
    
def dist_from_df(basedf,col="FEB",units="eV",plot=False):
    dist=[]
        
    done=[]
    lowbar=[]
    highbar=[]

    total=0
    skip=0
    subfolder="individual-dist/"
    
    
    df=basedf.dropna(subset=[col])
    
    #average over repeated runs, so any repeated pairs in a specific csv
    grpdf=df.groupby(["csvname","pair"],axis=0)[col].mean().reset_index()
    #print(grpdf.duplicated(subset=["csvname","pair"]).any())


    csvfiles=grpdf["csvname"].unique()
    setdf = pd.DataFrame({"csvname":csvfiles})
    cols=[f"{col}ratio",f"{col}Hnum",f"{col}mean",f"{col}stddev"]
    for c in cols:
        setdf[c]=""
        
    setdf.set_index("csvname", inplace=True)
    for csvfile in csvfiles:
        csdf=grpdf[grpdf["csvname"]==csvfile]

        print(csvfile)
        dist=csdf[col].tolist()
        
        (ratio,Hnum)=stats_from_csv_name(csvfile)
        setdf.at[csvfile,f"{col}ratio"]=ratio
        setdf.at[csvfile,f"{col}Hnum"]=Hnum
        setdf.at[csvfile,f"{col}mean"]=mean=statistics.mean(dist)
        setdf.at[csvfile,f"{col}stddev"]=stddev=statistics.stdev(dist)
        
        
        setdf[cols]=setdf[cols].apply(pd.to_numeric, errors='coerce')
        mstxt=f"Mean: {round(mean,2)}\nStd Dev: {round(stddev,2)}"
        total=len(csdf)

        
        num_bins=10
        N=len(dist)
        fig,ax = plt.subplots()
        ax.set_title(f'{col} Distribution for {total} runs using {str(csvfile).split("/")[-1]}')
        ax.set_xlabel(f'{col}({units})')
        ax.set_ylabel('Counts')
        ax.hist(dist,num_bins)
        
        #plot a proper sized normal distribution over the histogram
        bin_width = (max(dist) - min(dist)) / num_bins
        x = np.linspace(mean - 3*stddev, mean + 3*stddev, 100)
        ax.plot(x, stats.norm.pdf(x, mean, stddev)*N*bin_width)
        
        
        ax.text(0.01,0.99,mstxt,ha='left',va='top', transform=ax.transAxes, fontsize = 10)
        
        if plot:
            plt.show()
        else:
            
            name=csvfile.split('/')[-1].removesuffix('.csv')
            dirname=image_folder+subfolder+'/'+col+'/'
            os.makedirs(dirname, exist_ok=True)
            figname=dirname+name+'.svg'
            print(f"Saved figure with path {figname}")
            fig.savefig(figname)
            plt.close(fig)
            # print(f"Pairs that have FEB higher than 10: {highbar}")
            # print(f"Pairs that have FEB lower than 2.1: {lowbar}")
    
    #print(setdf)
    return setdf

def check_convergenc():
    #@TODO just a straight copy paste from jupyter, needs fixing before run
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import pandas
    import seaborn as sns
    import numpy as np

    #seaborn.set(style='ticks')


    base="/home/agoga/documents/code/topcon-md/output/neb/smallersample/"
    fileList=["125-133.csv","125-124.csv","125-134.csv","125-339.csv","125-126.csv","125-344.csv"]

    base="/home/agoga/documents/code/topcon-md/output/neb/sixloc/"
    fileList=["3924-1547.csv","3924-3632.csv","3924-1545.csv","3924-3599.csv","3924-1548.csv","3924-3955.csv"]

    base="/home/agoga/documents/code/topcon-md/output/neb/preciseconvergence"
    fileList=["3924-1547.csv"]

    base="/home/agoga/documents/code/topcon-md/output/neb/noHConvergence/"
    base="/home/agoga/documents/code/topcon-md/output/neb/HConvergence/"
    fileList=["pairs.csv"]

    #mirror pairs
    #133 126
    #344 134
    #339 124

    for f in fileList:
        d=base+f
        data = pandas.read_csv(d)
        #data = pandas.read_csv("/home/agoga/documents/code/topcon-md/output/neb/fixed/4090.csv")
        #data = pandas.read_csv("/home/agoga/documents/code/topcon-md/output/neb/varyTSmin/125.csv")



        data=data[data.etol<3e-5]
        data=data[data.etol>1e-06]
        data=data[data.ts>=0.5]
        data=data[data.ts<=0.6]
        data = data[data.FEB < 4]

        # data=data[data.etol <3e-5]
        # data=data[data.etol >3e-6]

        #sns.relplot(data=data,x="ts",y="A",hue="etol", kind="line",aspect=1.4,palette='tab10')#sns.color_palette("Set2"))
        sns.relplot(data=data,x="ts",y="FEB",hue="etol", kind="line",aspect=1.4,palette='tab10')#sns.color_palette("Set2"))

        #plt.hlines(y=3.2,color='r',xmin=0.3,xmax=2,linestyles='-.')
        # plt.ylim(top=3.6)
        plt.grid()
        plt.xticks(np.arange(0.5,0.7,0.1))
        plt.title('Forward energy barrier for different energy tolerances and timesteps')
    

def working_recursive_fun(df, og, path=[], debug=False):
    max_depth=5
    depth=len(path)
    if depth > max_depth:
        return [depth, path]
    
    curatom = og if depth == 0 else path[-1]
    
    curblist=df.at[curatom,'bonds']
    bonds=[item[0] for item in curblist]
    
    if debug:
        print(f'depth {depth} - Atom {curatom} has {bonds} bonded to it, path:{path}')
    
    if og in bonds and depth != 1:
        return [depth, path]
    
    minpath=None
    mindepth=max_depth
    for b in curblist:
        bi=b[0]
        newpath=path+[bi]
        
        ret=recursive_fun(df,og,newpath)
        
        if ret is not None and ret[0]< mindepth:
            mindepth=ret[0]
            minpath=ret[1]
     
    return [mindepth, minpath]


def double_recursive_fun(df, og,debug=False):
    
    ret1=recursive_fun(df,og)

    ret2=recursive_fun(df,og,[],ret1[1])

    if ret1[0]!=ret2[0]:
        print(f"{og} path1:{ret1} path2:{ret2}")
    return (ret1[0]+ret2[0])/2
    
        
def recursive_fun(df, og, path=[], badpaths=[],debug=False):
    max_depth=100
    depth=len(path)
    if depth > max_depth:
        return [depth, path]
    
    curatom = og if depth == 0 else path[-1]
    
    curblist=df.at[curatom,'bonds']
    bonds=[item[0] for item in curblist]
    
    if debug:
        print(f'depth {depth} - Atom {curatom} has {bonds} bonded to it, path:{path} badpath{badpaths}')
    
    if og in bonds and depth != 1:
        return [depth, path]
    
    minpath=None
    mindepth=max_depth
    for b in curblist:
        bi=b[0]
        newpath=path+[bi]
        
        ret=recursive_fun(df,og,newpath)
        
        if ret is not None and ret[0]< mindepth:
            good=True
            for b in badpaths:
                if ret[1]==badpaths:
                    good=False
            if good:
                mindepth=ret[0]
                minpath=ret[1]
     
    return [mindepth, minpath]

# from scipy.sparse import csr_matrix
# import networkx as nx

# def sparse_graph(df):     
#     csvlist=df["csvname"].unique()

#     for csvfile in csvlist:

#         print(csvfile)


#         dfc=df[df["csvname"]==csvfile]

#         (bonddf,box)=load_data_and_bonds_from_csv(csvfile)
        
#         numatoms=len(bonddf)
        
#         sg=nx.Graph()

#         for i in range(1,numatoms+1):
#             bonds=bonddf.at[i,"bonds"]
#             bl=[]
#             sg.add_node(i)
#             for b in bonds:
#                 if b[0] != i:
#                     sg.add_edge(i,b[0])
        
        
#         sg.add_edge(1,2)
#         sg.add_edge(2,3)
#         sg.add_edge(1,3)
#         # for s in sg.edges:
#         #     print(s)
        
#         #print(list(nx.edge_dfs(sg,5)))
        
#         tstart=time.time()
#         # for s in sg.nodes:
#         ti=time.time()
#         cycle=nx.find_cycle(sg,10,orientation='reverse')
#         # for start_node in sg.nbunch_iter(s):
#         #     print(start_node)
#         l=list(cycle)
#         tf=time.time()
#         print(f"){l} took {tf-ti}s")
            
            


#         return
            
# def get_path(Pr, i, j):
#     path = [j]
#     k = j
#     while Pr[i, k] != -9999:
#         path.append(Pr[i, k])
#         k = Pr[i, k]
#     return path[::-1]


# def find_rings(g,ring_size):

#     D,Pr =csg.shortest_path(g,directed=False, return_predecessors=True)
    
    
#     for r in range(1,4):
#         inds=list(zip(*np.where(D==r)))

#         for i in inds:
#             print(get_path(Pr,i[0],i[1]))
        
        

if __name__ == "__main__":       
        
    # nebfolder="/home/agoga/documents/code/topcon-md/neb-out/"


    # group=["FullSet/"] #"farm/6169019/","farm/6169020/","farm/6169021/","farm/6169022/"]#
    # #group=["farm/perpPairs/","farm/parallelPairs/"]

    # multiplot=[]


    # for g in group:
    #     csvlist=[]
    #     gpath=nebfolder+g
    #     csvlist=[]
    #     for d in Path(gpath).glob('*.csv'):
    #         csvlist.append(str(d))
        
    #     dstats=[]
    #     #setdf=csv_to_df(csvlist[0])
    #     #df=dist_from_df(setdf,False)
    #     csvname=csvlist[0].split('/')[-1].removesuffix('.csv')
        
    #     (posdf,box)=load_datafile_from_csv(csvname)

    #     depthlist=[]
    #     for index, row in posdf.iterrows():
    #         if row['type']=='O':
    #             path=double_recursive_fun(posdf,index)
    #             if path is not None:
    #                 depthlist.append(path)

    #             # path=recursive_fun(posdf,index)
    #             # if path is not None:
    #             #     depthlist.append(len(path))
            
        
    #     print(np.mean(depthlist))
    #     print(np.min(depthlist))
    #     print(np.max(depthlist))
    #     #nt.plot_multi_distribution(df)
        
    #     #rangeddf =plot_atominrange(setdf,10,'H')
    
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    import statistics
    from pathlib import Path
    from operator import itemgetter


    pairspath="/home/agoga/documents/code/topcon-md/data/neb/FullSet/"
    nebfolder="/home/agoga/documents/code/topcon-md/neb-out/"


    group=["PinholeCenterZap/"] #"farm/6169019/","farm/6169020/","farm/6169021/","farm/6169022/"]#
    #group=["farm/perpPairs/","farm/parallelPairs/"]

    multiplot=[]
    clean=False #do this very rarely when you get lots of new data from farm


    for g in group:
        
        gpath=nebfolder+g
    
        print(gpath)
        
        
        if clean:
            cleanlist=[] 
            subfolders="farm-folders/"
            subfolders=""   
            print(f"Cleaning the {g+subfolders} directory, this may take a while.")
            for d in Path(gpath+subfolders).rglob('**/*.csv'):
                cleanlist.append(str(d))        
            cleandf=clean_csvs(cleanlist,gpath)
            
            #nt.clean_pairfiles(cleandf,pairspath)
        else:
            csvlist=[]
            i=0
            for d in Path(gpath).glob('*.csv'):
                csvlist.append(str(d))
                # if i >1:
                #     break
                # i+=1
                

            #print(csvlist[0])
            setdf=csvs_to_df(csvlist)
            
    pair_path="/home/agoga/documents/code/topcon-md/data/neb/PinholeCenterZap/"

    t=calc_local_structure(setdf,pair_path) #,["REB"])