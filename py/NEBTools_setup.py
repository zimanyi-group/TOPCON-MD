#!/usr/bin/env python
"""
Author: Adam Goga
This script contains functions that are primarily used during the initial setup of the NEB pipeline either in the PrepNEB.py file or the 
CreatePariList.py file. Some functions also needed for post pipeline analysis, may be defined here. This script also contains a good deal
of geometric functions which are designed specifically to work on systems with periodic boundaries.
"""
import os
import matplotlib.pyplot as plt

from sympy import Point3D, Plane

import numpy as np

from lammps import lammps
import re

import pandas as pd
pd.options.mode.chained_assignment = None 

import os
from mpi4py import MPI
import time

from ase.geometry import get_angles
import ase.cell

from pandas.api.types import is_numeric_dtype
from numpy.linalg import norm

jp=0


me = MPI.COMM_WORLD.Get_rank()
numproc=MPI.COMM_WORLD.Get_size()

#conversion from kcal/mol to eV
conv=0.043361254529175



datafolder="/home/adam/code/topcon-md/data/neb/"



def pbc_midpoint(simbox,p1,p2):
    """
    Calculate the midpoint between two points considering periodic boundary conditions.
    :param simbox - the simulation box dimensions
    :param p1 - the first point
    :param p2 - the second point
    :return The midpoint coordinates considering periodic boundary conditions.
    """
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

def pbc_vec_subtract(simbox, posi,posf,normalize=False):
    """
    Calculate the vector difference between two positions considering periodic boundary conditions.
    :param simbox - the simulation box dimensions
    :param posi - initial position vector
    :param posf - final position vector
    :return The vector difference accounting for periodic boundary conditions.
    """
    ret=[0,0,0]
    for i, (a, b) in enumerate(zip(posi, posf)):
        delta = abs(b - a)
        dimension=simbox[i,1]-simbox[i,0]
        if delta > dimension - delta:
            if b < a:
                b+=dimension
            else:
                a+=dimension
            
        ret[i]=b-a
        
    if normalize:
        norm=ret / np.linalg.norm(ret)
        return norm
    return ret


def vec_addition(v1,v2,normalize=False):
    """
    Perform vector addition on two input vectors and optionally normalize the result.
    :param v1 - The first vector
    :param v2 - The second vector
    :param normalize - A boolean flag indicating whether to normalize the result (default is False)
    :return The result of vector addition, optionally normalized
    """
    ret=[0,0,0]
    for i, (a, b) in enumerate(zip(v1, v2)):
        ret[i]=a+b
        
    if normalize:
        norm=ret / np.linalg.norm(ret)
        return norm
    return ret

def pbc_position_correction(simbox,pos):
    """
    Correct the position of a particle based on the periodic boundary conditions of the simulation box.
    :param simbox - the simulation box dimensions
    :param pos - the current position of the particle
    :return The corrected position of the particle within the simulation box.
    """
    ret=[0,0,0]
    
    for i, p in enumerate(pos):
        dimension=simbox[i,1]-simbox[i,0]
        
        if p < simbox[i,0]:
            ret[i]=p+dimension
        elif p>simbox[i,1]:
            ret[i]=p-dimension
        else:
            ret[i]=p
    return ret
        

def pbc_add_vec_point(simbox, point,vec):
    ret=[0,0,0]
    for i, (a, b) in enumerate(zip(point, vec)):
        c= a+b
        max_dim=simbox[i,1]
        min_dim=simbox[i,0]
        dim_len=max_dim-min_dim
        if c > max_dim:
            c -= dim_len 
        if c < min_dim:
            c += dim_len
            
        ret[i]=c
        
    return ret

def pbc_dist(simbox, pos1,pos2):
    """
    Calculate the minimum image convention distance between two positions in a periodic boundary condition simulation box.
    :param simbox - the simulation box dimensions
    :param pos1 - position of the first object
    :param pos2 - position of the second object
    :return the minimum image convention distance between pos1 and pos2
    """
    total = 0
    for i, (a, b) in enumerate(zip(pos1, pos2)):
        delta = abs(b - a)
        dimension=simbox[i,1]-simbox[i,0]
        if delta > dimension- delta:
            delta = dimension - delta
        total += delta ** 2
    return total ** 0.5


def vec_projection(v1,v2):
    """
    Calculate the projection of vector v1 onto vector v2.
    :param v1 - The first vector
    :param v2 - The second vector
    :return The scalar projection of v1 onto v2
    """
    return np.dot(v1,v2)/np.linalg.norm(v2)

def vec_proj_to_plane(v1,v2):
    """
    Calculate the projection of vector v1 onto the plane defined by vector v2.
    :param v1 - The vector to be projected.
    :param v2 - The vector defining the plane.
    :return The projection of v1 onto the plane defined by v2.
    """
    s_proj=vec_projection(v1,v2)
    return v1-s_proj*v2
    
def gen_points_sphere(center,r,num_pts,plot=False):
    """
    Generate points on a sphere given a center, radius, and number of points.
    :param center - the center of the sphere
    :param r - the radius of the sphere
    :param num_pts - the number of points to generate
    :param plot - whether to plot the points (default is False)
    :return a list of points on the sphere
    """
    
    num_split=int(num_pts**(1/2))

    
    x0=center[0]
    y0=center[1]
    z0=center[2]
    pts=[]
    if plot:
        ax = plt.figure().add_subplot(projection='3d')
    for theta in np.linspace(0,2*np.pi,num_split,endpoint=False):
        for j in np.linspace(-1,1,num_split):
            phi=np.arccos(j)
            
            x = x0+r*np.cos(theta)*np.sin(phi)
            y = y0+r*np.sin(theta)*np.sin(phi)
            z = z0+r*np.cos(phi)
            pts.append([x,y,z])
            if plot:
                ax.scatter(x,y,z)
    if plot:
        plt.show()
    return np.array(pts)

    
def pbc_dist_point_to_vec(simbox, p1,p2,distPoint):
    """
    Calculate the minimum distance between a point and a line segment in a periodic boundary condition simulation box.
    :param simbox - the simulation box dimensions
    :param p1 - the first point of the line segment
    :param p2 - the second point of the line segment
    :param distPoint - the point for which the distance is calculated
    :return The minimum distance between the point and the line segment
    """
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
    """
    Apply the point-vector distance calculation to a DataFrame.
    :param df - The DataFrame containing the data
    :param simbox - The simulation box
    :param p1 - Point 1
    :param p2 - Point 2
    :param atomtype - Type of atom (optional)
    :param col - Column name (default is 'pos')
    :return A tuple containing the updated DataFrame and the name of the new column.
    """
    pvdcol='pointvecDist'
    if df.empty:
        return df
    distdf=df.copy()
    
    if atomtype is not None:
        distdf=distdf[distdf["type"]==atomtype]

    distdf[pvdcol]=distdf.apply(lambda row: pbc_dist_point_to_vec(simbox,p1,p2,row[col]),axis=1)

    return (distdf,pvdcol)
           

def apply_dist_from_pos(df,simbox,pos,atomtype=None,col='pos'):
    """
    Apply the distance calculation between the given column 'col' in a DataFrame and a specified position.
    :param df - The DataFrame containing positions.
    :param simbox - The simulation box dimensions.
    :param pos - The position to calculate distances from.
    :param atomtype - The type of atom to consider (optional).
    :param col - The column containing positions in the DataFrame (default is 'pos').
    :return A DataFrame with an additional 'dist' column containing the calculated distances.
    """
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
    Hstr=filename.split('-')[1]
    Hnum=Hstr.split('_')[0]
    #print(f"ratio: {ratio}, Hnum:{Hnum}")
    return (ratio,Hnum)

def angle_between_vec(box,v1,v2,debug=False):
    """
    Use ASE to calculate the angle between two vectors in a given simulation box
    """
    cell=ase.cell.Cell([[box[0,0],box[1,1],box[2,1]],[box[0,1],box[1,0],box[2,1]],[box[0,1],box[1,1],box[2,0]]])
    ang=get_angles([v1],[v2],cell,True)

    return (ang,[v1,v2])


def angle_between_pts(box,p1, p2,pm,debug=False):
    """
    Calculate the angle between two points and a middle point.
    :param box - the box dimensions
    :param p1 - point 1
    :param p2 - point 2
    :param pm - middle point
    :param debug - flag for debugging
    :return A tuple containing the angle between the two vectors and the two vectors 
    """
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

def angle_in_plane(box,plane_point,plane_vec,orig_point,plane_zero_vec,ax=None):
    """
    Project two points onto a plane where one of the points is used to create the plan
    :param box - The simulation box for PBC
    :param plane_point - A point on the plane
    :param plane_vec - The normal vector of the plane
    :param orig_point - The point to create a vector that is then projected onto the plane
    :param plane_zero_vec - The zero vector of the plane
    :param ax - Optional parameter for plotting
    :return The angle between the point and the zero vector on the plane.
    """
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


def find_minima_around_atom(box,file,move_atom,final_atom,neighbor_atoms):
    '''
    Find lowest energy configuration for moving the move_atom to a location around the final_atom
    
    Make sure not to test locations near other atoms to speed this up.
    '''
    from sympy.abc import u,v
    
    
    
    ti=time.time()

    L = lammps('mpi',["-log",'none',"-screen",'none'])
    init_dat(L,file)
    
    L.commands_string(f'''
        variable x{move_atom} equal x[{move_atom}]
        variable y{move_atom} equal y[{move_atom}]
        variable z{move_atom} equal z[{move_atom}]
        ''')
    
    L.commands_string(f'''
        variable x{final_atom} equal x[{final_atom}]
        variable y{final_atom} equal y[{final_atom}]
        variable z{final_atom} equal z[{final_atom}]
        ''')
    
    xi = L.extract_variable(f'x{move_atom}')
    yi = L.extract_variable(f'y{move_atom}')
    zi = L.extract_variable(f'z{move_atom}')
    
    x_tester = L.extract_variable(f'x{final_atom}')
    y_tester = L.extract_variable(f'y{final_atom}')
    z_tester = L.extract_variable(f'z{final_atom}')
    
    orig=np.array([xi,yi,zi])
    
    Ef=0
    Ei = L.extract_compute('thermo_pe',0,0)*conv
    minE=Ei
    

def find_bond_preference(box,file,atom,midpoint,sepvec):
    """
    This function is intended to find the bond preference of an atom based on various parameters.
    :param box - the box parameter
    :param file - the file parameter
    :param atom - the atom parameter
    :param midpoint - the midpoint of the final BC
    :param sepvec - the separation vector of the atom and the final BC
    :return the lowest energy point nearby that the atom prefers to be bonded at 
    """
    #@TODO ADAM this is one of functions to make work for H
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

def find_local_minima_position(file,atom,initial_guess):
    """
    Find the position of a local minima for a given atom in a file using an initial guess.
    :param file - the lammps data file
    :param atom - the atom for which the local minima position needs to be found
    :param initial_guess - the initial guess for the position
    :return the lowest energy position of the atom given the system in the inputs
    """
    #@TODO ADAM this is one of the functions to do for H
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


def find_nearby_atoms(file,atom,radius):
    """
    Find all atoms near a specified atom within a given radius in a LAMMPS simulation.
    :param file - the LAMMPS input file
    :param atom - the atom to find nearby atoms for
    :param radius - the radius within which to search for nearby atoms
    :return all atoms within the radius of a given atom including that atom
    """

    L1 = lammps('mpi',["-log",'none',"-screen",'none'])
    init_dat(L1,file)

    i_pos=find_atom_position(L1,atom)
    
    
    L1.commands_string(f'''
    region r_local sphere {i_pos[0]} {i_pos[1]} {i_pos[2]} {radius}
    group g_local region r_local

    variable local_i atom id
    variable local_type atom type
    # variable local_x atom x
    # variable local_y atom y
    # variable local_z atom z
    ''') 
    
    
    ids     = np.array(L1.extract_variable("local_i","g_local"))
    types   = np.array(L1.extract_variable("local_type","g_local"))
    # xs      = np.array(L1.extract_variable("local_x","g_local"))
    # ys      = np.array(L1.extract_variable("local_y","g_local"))
    # zs      = np.array(L1.extract_variable("local_z","g_local"))
    
    
    id_list=ids[ids!=0]
    types=types[types!=0]
    # x_list=xs[xs!=0]
    # y_list=ys[ys!=0]
    # z_list=zs[zs!=0]
    
    # print(f"{len(id_list)}-{len(types)}-{len(x_list)}-{len(y_list)}-{len(z_list)}")
    # final_list=[]
    # for i in range(len(id_list)):
    #     id=id_list[i]
    #     type = types[i]
        
    #     if id == atom:
    #         continue
        
    #     if type == 1:
    #         type='Si'
    #     elif type == 2:
    #         type='O'
    #     elif type == 3:
    #         type ='H'
            
    #     final_list.append([id,type])
        
        # print(f"{id_list[i]}-Type {types[i]}")
    
    return np.array(id_list)
    

def find_empty_space(simbox,center,atom_positions,grid_spacing=.1,atom_spacing=4,min_dist=2,plot=False):
    """
    Find an empty space in a simulation box given the center of a region to look at, atom positions, and other parameters.
    :param simbox - the simulation box dimensions
    :param center - the center point to find empty space around
    :param atom_positions - positions of atoms in the simulation box
    :param grid_spacing - spacing of the grid that ultimately finds the empty space
    :param atom_spacing - required distance that the empty space must be from all other atoms
    :param min_dist - minimum distance between the center and the empty space, no space will be returned closer than this distance
    :param plot - whether to plot the results
    :return The coordinates of the empty space we have found
    """
    if plot:
        ax = plt.figure().add_subplot(projection='3d')
    max_dist=0
    
    
    for atom in atom_positions:
        apos=atom[1]
        dist=pbc_dist(simbox,center,apos)
        if max_dist<dist:
            max_dist=dist
    
    
    tot_spaces=int(2*np.ceil(max_dist/grid_spacing)+1)
    middle=int(np.ceil(max_dist/grid_spacing))
    num_grids_to_fill=int(np.ceil(atom_spacing/(2*grid_spacing)))
    min_dist=min_dist/grid_spacing
    
    #print(f"num_grids_to_fill-{num_grids_to_fill}")
    #print(f"max_distance is {max_dist}, middle of grid - {middle} with {tot_spaces} total spaces")


    grid=np.zeros((tot_spaces,tot_spaces,tot_spaces))

    # print(f"{tot_spaces} - {middle}")
    for atom in atom_positions:
        apos=atom[1]
        diff=pbc_vec_subtract(simbox,center,apos)
        xi=middle+int(round(diff[0]/grid_spacing))
        yi=middle+int(round(diff[1]/grid_spacing))#minus because +y up is convention but a lower index is 'up
        zi=middle+int(round(diff[2]/grid_spacing))
        # print(f"{atom[0]} - {xi},{yi},{zi}")
        if plot:
            ax.scatter(xi-middle,yi-middle,zi-middle,marker='o',c='b')
        
        
    
        # 
        # if plot:
        #     u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        #     x = xi-middle+np.cos(u)*np.sin(v)*num_grids_to_fill
        #     y = yi-middle+np.sin(u)*np.sin(v)*num_grids_to_fill
        #     z = zi-middle+np.cos(v)*num_grids_to_fill
        #     ax.plot_wireframe(x, y, z, color="r")
            

        
        
        #fill the grid for each atom with some space around 
        for i in np.arange(xi-num_grids_to_fill,xi+num_grids_to_fill+1,step=1):
            for j in np.arange(yi-num_grids_to_fill,yi+num_grids_to_fill+1,step=1):
                for k in np.arange(zi-num_grids_to_fill,zi+num_grids_to_fill+1,step=1):
                    # print(k)
                    # fill_x= xi+i
                    # fill_y= yi+j
                    # fill_z= zi+k
                    # if not (i == xi-num_grids_to_fill or i==xi+num_grids_to_fill):
                    #     continue
                    # if not (j == yi-num_grids_to_fill or j==yi+num_grids_to_fill):
                    #     continue
                    # if not (k == zi-num_grids_to_fill or k==zi+num_grids_to_fill):
                    #     continue
                    
                    fill_x = i if i < tot_spaces else tot_spaces - 1
                    fill_y = j if j < tot_spaces else tot_spaces - 1
                    fill_z = k if k < tot_spaces else tot_spaces - 1
                    
                    grid[fill_x,fill_y,fill_z]=1
    if plot:
        # u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        # x = np.cos(u)*np.sin(v)*num_grids_to_fill
        # y = np.sin(u)*np.sin(v)*num_grids_to_fill
        # z = np.cos(v)*num_grids_to_fill
        # ax.plot_wireframe(x, y, z, color="b")
        
        ax.scatter(0,0,0,marker='o',c='r')
        
    #after all the atoms have filled the grid, now find the closest empty space
    cur_min_dist=tot_spaces
    cur_min_pos=[-1,-1,-1]
    for x in range(len(grid)):
        for y in range(len(grid[x])):
            for z in range(len(grid[x,y])):
                if grid[x,y,z] == 0:
                    dist=((x-middle)**2+(y-middle)**2+(z-middle)**2)**(1/2)
                    if dist < cur_min_dist and dist > min_dist:
                        # print('min found')
                        cur_min_dist=dist
                        cur_min_pos=[x-middle,y-middle,z-middle]
                    
                    # ax.scatter(x-middle,y-middle,z-middle,marker='o',c='r')
                    
    #now convert to real coordinates
    #print(f"min_pos {cur_min_pos}")
    min_pos_vector=grid_spacing*np.array(cur_min_pos)#negative cause i'm bad and only have vector subtract.....
    final_coords=pbc_add_vec_point(simbox,center,min_pos_vector)
    
    #print(f"Min_pos_vector {min_pos_vector}")
    #print(f"simbox: {simbox} center: {center}")
    #print(f'Closest empty space - {final_coords[0]} {final_coords[1]} {final_coords[2]}')
    
    if plot:
        ax.scatter(cur_min_pos[0],cur_min_pos[1],cur_min_pos[2],marker='x',c='g')   
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
    
    return final_coords

def test_location_pe(file,atom,r_i,log=False):
    '''
        r_i - initial guess
        
        returns 
            r[0] = final positoon
            r[1] = delta_PE
    '''
    #@TODO ADAM this is one of the functions to fuck around with

    #Need two lammps instances so that when removing an atom and minimizing we don't increase time for final NEB image minimization
    if log:
        L = lammps('mpi',["-log",'none'])
    else:
        L = lammps('mpi',["-log",'none',"-screen",'none'])
    init_dat(L,file)

    Ei = L.extract_compute('thermo_pe',0,0)*conv
    
    
    
    
    L.commands_string(f'''
                    set atom {atom} x {r_i[0]} y {r_i[1]} z {r_i[2]}
                    run 0
                    ''')

    NEB_min(L,7e-6)
    
    Ef= L.extract_compute('thermo_pe',0,0)*conv
    rf=find_atom_position(L,atom)
    
    
        
    return [rf, (Ef-Ei)]

def test_multi_location_pe(file,atom,r_l,log=True):
    '''
        r_i - list of initial guess
        
        returns 
            r[0] = final positoon
            r[1] = delta_PE
    '''

    #Need two lammps instances so that when removing an atom and minimizing we don't increase time for final NEB image minimization
    if log:
        L = lammps('mpi',["-log",'none'])
    else:
        L = lammps('mpi',["-log",'none',"-screen",'none'])
    init_dat(L,file)

    Ei = L.extract_compute('thermo_pe',0,0)*conv
    
    ret=[]
    print(f"Testing {len(r_l)} locations")
    i=0
    for r_i in r_l:
        print(f"{i}/{len(r_l)}")
        L.commands_string(f'''
                        set atom {atom} x {r_i[0]} y {r_i[1]} z {r_i[2]}
                        run 0
                        ''')

        NEB_min(L,7e-6)
        
        Ef= L.extract_compute('thermo_pe',0,0)*conv
        rf=find_atom_position(L,atom)
        
        ret.append([rf, (Ef-Ei)])
        i+=1
        
    return np.array(ret,dtype=object)

def discover_dumpsteps(dump_file):
    '''
    return a list of what the dump steps in this dump file are
    '''

    with open(dump_file,'r') as f:
        file=f.read()
    
    match = re.findall("(ITEM: TIMESTEP\n(.*))",file)
    
    dump_steps = np.array([int(element[1]) for element in match])
        
    return dump_steps
        

def create_bond_file(datapath, file,filename,dumpstep=None):
    """
    Create a ReaxFF bond information file using LAMMPS simulation software. Creates and kills a LAMMPS simulation using the python wrapper.
    :param datapath - the path to the data
    :param file - the file to be processed
    :param bondfile - the bond file to be created
    """
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
        ''')
    
    if  dumpstep is not None:
        L.commands_string(f'''
        region sim block 0 1 0 1 0 1
        
        
        lattice diamond 5.43

        create_box 3 sim

        read_dump {datapath+file} {dumpstep} x y z box yes add keep
        ''')
    else:
        L.commands_string(f'''
        read_data {datapath+file}
        ''')
        
    L.commands_string(f'''
        
        mass         3 $(v_massH)
        mass         2 $(v_massO)
        mass         1 $(v_massSi)


        min_style quickmin
        
        pair_style	    reaxff /home/adam/code/topcon-md/potential/topcon.control
        pair_coeff	    * * /home/adam/code/topcon-md/potential/ffield_Nayir_SiO_2019.reax Si O H

        neighbor        2 bin
        neigh_modify    every 10 delay 0 check no
        
        thermo $(v_printevery)
        thermo_style custom step temp density press vol pe ke etotal #flush yes
        thermo_modify lost ignore
        
        
        
        fix r1 all qeq/reax 1 0.0 10.0 1e-6 reaxff
        
        # reset_atom_ids //this breaks things...
        
        compute c1 all property/atom x y z
        fix f1 all reaxff/bonds 1 {datapath}/scratchfolder/{filename}.bonds
        
        run 0
        ''')
    
    if dumpstep is not None:
        L.commands_string(f'''
        write_data {datapath}/scratchfolder/{filename}.data
            ''')
    
def read_bonds(df,file):
    """
    Read bond information from a file and update a DataFrame with the bond data.
    :param df - The DataFrame to update with bond information.
    :param file - The file containing bond information.
    :return Dataframe with bond information
    """
    # alist=[]
    #create new columns in the datafram
    df["bonds"]=pd.Series(dtype=object)
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
                print(f'Big bombad making bonds - Atom{id} doesnt match types: {df.at[id,"type"]} != {typ}')
                
            df.at[id,'nb']=nb
            i=3
            blist=[]
           
           #run through each bond and grab the id and bond order(index offset by the number of bonds)
            for j in range(nb):
                blist.append([int(l[i+j]),float(l[i+j+nb+1])])
            #print(blist)
            # df.at[id,"bonds"]=""
            df.at[id,"bonds"]=blist

    return df

def read_data(file):
    """
    Read data from a LAMMPS data file and store it in a DataFrame.
    :param file - the file to read the data from
    :return a DataFrame containing the data
    """
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

def read_dump(dumpfile,dumpstep):
    """
    Read a LAMMPS dump file and store it in a DataFrame.
    :param dumpfile - the file to read the data from
    :param dumpstep - the dumpstep to load 
    :return a DataFrame containing the data
    """
    dlist=[]
    skipindex=0
    simbox=np.zeros((3,2))
    
    timestep_pattern = f"(?s)ITEM: TIMESTEP\n{str(int(dumpstep))}\n(.*?)(?=ITEM: TIMESTEP)"

    box_pattern = "(?s)ITEM: BOX BOUNDS pp pp pp\n(.*?)\n(?=ITEM: ATOMS id type xs ys zs)"

    atoms_pattern = "(?s)ITEM: ATOMS id type xs ys zs\n(\n|.*)"
    with open(dumpfile,'r') as f:
        file=f.read()
        
    atom_data = re.findall(timestep_pattern,file)[0]
    # print(ts_match)
    # print(match[0][1])
    # atom_data=""
    # for m in ts_match:
    #     if int(m[0]) == int(dumpstep):
    #         atom_data=m[1]
    
    # print(atom_data)
    # print(atom_data)
    box_match=re.findall(box_pattern,atom_data)

    box=box_match[0].split('\n')
    for bi in range(len(box)):
        simbox[bi][0]=box[bi].split(' ')[0]
        simbox[bi][1]=box[bi].split(' ')[1]
        
    atom_match=re.findall(atoms_pattern,atom_data)[0]

    for line in atom_match.split('\n'):
        if line == '':
            continue
        el = line.split(' ')

        
        id=int(el[0])
        typ=int(el[1])
        if typ == 1:
            typ='Si'
        elif typ==2:
            typ='O'
        elif typ==3:
            typ='H'
        x=float(el[2])
        y=float(el[3])
        z=float(el[4])
        pos=np.array([x,y,z])
        q=0
        dinfo=[id,typ,q,x,y,z,pos]
            
        #add this row to the dataframe
        dlist.append(dinfo)


    df = pd.DataFrame(dlist,columns=['id','type','q','x','y','z','pos'])
    df.set_index('id',inplace=True,drop=True)

    return (df,simbox)

def get_lammps(log):
    return lammps('mpi',["-log",log,'-screen','none'])

def init_dat(L,file):
    """
    Initialize a data file for simulation with specific parameters.
    :param L - The LAMMPS python wrapper simulation object
    :param file - The file to read data from
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


def read_file_data_bonds(datapath,dfile,dumpstep=None):
    """
    Read my custom data from a lammps data file and my custom made bond file then extract atom information.
    :param datapath - the path to the data files
    :param dfile - the data file to read
    :return a DataFrame containing all the atoms and their bond information and simulation box dimensions
    """
    

    filename=dfile.removesuffix('.dat').removesuffix('.data').removesuffix('.dump')


    os.makedirs(datapath+'/scratchfolder/', exist_ok=True)

    if dumpstep is None:
        
        create_bond_file(datapath,dfile,filename)
        (dfdata,simbox)=read_data(datapath+dfile)
    else:
        create_bond_file(datapath,dfile,filename,dumpstep)
        (dfdata,simbox)=read_data(datapath+'/scratchfolder/'+filename+'.data')
        
    
    df=read_bonds(dfdata,datapath+'/scratchfolder/'+filename+".bonds")

    return (df,simbox)

def find_all_in_range(atoms, simbox,atom,radius):
    """
    Find all atoms within a certain radius of a specified atom in a simulation box.
    :param atoms - DataFrame containing information about all atoms
    :param simbox - Simulation box dimensions
    :param atom - Atom of interest
    :param radius - Radius within which to search for other atoms
    :return Array of atom IDs within the specified radius of the given atom
    """
    center=atoms.loc[atom,'pos']
    
    distdf=apply_dist_from_pos(atoms,simbox,center)
    distdf=distdf[distdf['dist']<radius]
    atomids=distdf.index.to_numpy()
    return atomids

def find_movers_neighbor(df,curatom,zappdatom,natom="Si",nnatom="O"):
    """
    Find the neighboring atom that is connected to a specific atom in a dataframe.
    :param df - the dataframe containing atom information
    :param curatom - the current atom index
    :param zappdatom - the atom to which the current atom is connected
    :param natom - the type of atom to search for (default is "Si")
    :param nnatom - the type of neighboring atom to search for (default is "O")
    :return The index of the neighboring atom or None if not found
    """
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
    """
    Find suitable neighbors for a given atom based on certain criteria.
    :param df - the dataframe containing atom information
    :param curatom - the current atom for which we are finding neighbors
    :param zappdatom - the atom to be avoided
    :param natom - the type of atom to consider as neighbors (default is "Si")
    :param nnatom - the type of atom to avoid as neighbors (default is "O")
    :return The suitable neighbor for the current atom
    """
    ni=find_movers_neighbor(df,curatom,zappdatom,natom,nnatom)
    if ni is None:
        return 0
    
    curbonds=df.at[curatom,'bonds']
    neibonds = df.at[ni,'bonds']
    for b in curbonds:
        if b[0] == ni:
            return b[1]


def find_nnneighbor(df,curatom,zappdatom,natom="Si",nnatom="O"):
    """
    Find the next-nearest neighbor of a given atom in a dataframe based on specified criteria.
    :param df - the dataframe containing atom information
    :param curatom - the current atom index
    :param zappdatom - the atom to be compared for nearest neighbor
    :param natom - the type of atom to search for (default is "Si")
    :param nnatom - the type of nearest neighbor atom to search for (default is "O")
    :return The index of the nearest neighbor atom or None if not found.
    """
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

def find_neighboring_sibc(atoms,oi):
    """
    Find a neightboring oxygen vacancy i.e. silicon-silicon bond center
    :param atoms - the atoms df object
    :param oi - the oxygen atom to check neighboring si's of
    :return The final pair of silicon atoms that the atom ended up inbetween
    """
    #start with an Oxygen and find all neighboring Si with BC vacancies
    nindices=atoms.at[oi,'bonds']
    #look to see if this atom has any neighboring SI with vacancies

    si_bc_vac=[]
    badsi=[]
    for n in nindices:
        
        ni=n[0]
        neitype=atoms.at[ni,'type']
        if neitype=='Si':

            sibonds=atoms.at[ni,'bonds']
                    
            badsi=[]  
            for b in sibonds:
                bi=b[0]
                btype=atoms.at[bi,'type']
                if btype=='H' or btype=='O':
                    bb=atoms.at[bi,'bonds']
                    for bn in bb:
                        bni=bn[0]
                        bntype=atoms.at[bni,'type']
                        if bntype=='Si' and bni !=ni:
                            badsi.append(bni)
        
            
            for nnn in sibonds:
                nnni=nnn[0]
                nnntype=atoms.at[nnni,'type']
                if nnntype=='Si'and nnni not in badsi:
                    si_bc_vac.append([ni,nnni])
    
    return si_bc_vac


if __name__ == "__main__":       
    do_nothing=0