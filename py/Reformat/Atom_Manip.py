#original file written by Andrew Diggs
import ReadLAMMPS as rl
import atom
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import pdb
import math


NA = 6.022e23


def Plot_Hist(dat):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    ax.bar3d(dat[:,0], dat[:,1],0.0, 5.3,5.3, dat[:,2])
    
    plt.show()

def region(atom):
    d=5.43
    reg=np.empty([3,2])
    for i in [0,1,2]:
        low, high = atom.coords[i] - d, atom.coords[i] + d
        if(low < atom.box[i][0]):
            low=atom.box[i][0]
        elif(high > atom.box[i][1]):
            high=atom.box[i][1]
        reg[i] = [low,high]
    return reg


def get_env(at):
    i=0
    nlist=[]
    simask=[]
    for n in at.neighbors:
        nlist.append(atom.distance(at,n))
        simask.append(n.type==2)
    nlist=np.asarray(nlist) #array of at neighbor distances
    simask=np.asarray(simask) #Bool array of at silicon neighbors
    bmask=np.logical_and(nlist>1.3,nlist<1.6) #bool array for is at bonded to a Si
    nnmask=np.logical_and(nlist>2.0, nlist<2.8) #bool array for how many neighbors are 2.4->2.8 A away
    sinn=np.logical_and(simask,nnmask) #bool array for are Si neighbors between 2.4->2.8 
    bsi=np.logical_and(simask,bmask) #bool array for is the neighbor 1.4->1.6 a Si atom
    SiBonded=(np.count_nonzero(bsi)==1) #count to make sure only one Si neighbor is 1.4->1.6, if more than 1
    # then H atom is at a bond center.
    count_nn = np.count_nonzero(sinn)
    if(SiBonded and count_nn >= 1):
        nn = at.neighbors[sinn]
        return move_H(at, nn, count_nn)

    else:
        return np.asarray([at])

def move_H(at,nn, cnt):
    if(cnt >= 2):
        for n in nn:
            for oth in nn:
                if(n.id != oth.id):
                    dist=atom.distance(n,oth)
                    if(dist > 2.15 and dist < 2.8):
                        return move_bc(at,n,oth)
    #if no nn are also neighbors then move H to an interstitial site away from the two si atoms.
    return move_inter(at,nn)

def move_bc(at, n1, n2):
    print("BC")
    r=n1.coords - n2.coords
    r_hat=r/np.sqrt(np.sum(r*r))
    mid=(n1.coords + n2.coords)/2.0
    at.coords=mid
    n1.coords= n1.coords + 0.45*r_hat
    n2.coords= n2.coords - 0.45*r_hat
    at.distance_moved()
    n1.distance_moved()
    n2.distance_moved()
    return np.array([at,n1,n2])

def move_inter(at, nn):
    s=np.size(nn)
    if(s >= 2):
        r1= nn[0].coords - at.coords
        r2= nn[1].coords - at.coords
        r_vec= r1 + r2
        r_hat=r_vec/np.sqrt(np.sum(r_vec*r_vec))
        at.coords = at.coords + 3.35*r_hat
        print("2 Neighbors")
        at.distance_moved()
        return np.asarray([at])
    elif(s==1):
        r_vec= nn[0].coords - at.coords
        r_hat=r_vec/np.sqrt(np.sum(r_vec*r_vec))
        at.coords = nn[0].coords + 1.5*r_hat
        print("1 Neighbor")
        at.distance_moved()
        return np.asarray([at])
    else:
        return np.asarray([at])

def get_hydrogen(array_of_atoms):
    mask=np.empty([np.size(array_of_atoms)],dtype=bool)
    i=0
    cont=0
    for at in array_of_atoms:
        if(at.type == 1):
            mask[i]=True
            i+=1
            cont+=1
        elif(at.type == 2):
            mask[i]=False
            i+=1
    return array_of_atoms[mask]

def Migration(at, file):
    print_env(at)
    moved=get_env(at)
    final = io.write_final(moved, at.id, file)
    if not(final):
        print("Hydrogen atom {0} was not moved")
        return 1
    else:
        io.comp(file,final)
        return 0

def print_env(at):
    atype=["","H","Si"]
    print("")
    txt1="{0} {1}: {2:.2f} {3:.2f} {4:.2f}".format(atype[at.type], at.id, *at.coords)
    txt2="{0} {1} {2:.3f}"
    print(txt1)
    print("Type: ID: Distance:")
    for n in at.neighbors:
        ftext=txt2.format(atype[n.type], n.id, atom.distance(at,n))
        print(ftext)
    return 2



def Compute_Density(array_of_atoms):
    Atom_Masses = [0.0, 1.008, 28.0855]
    mass = []
    cut = 9.8
    vol = atom.Compute_Volume(cut)
    for at in array_of_atoms:
        if(at.coords[2] > cut):
            mass.append(Atom_Masses[at.type])
    num = len(mass)
    M_tot = np.sum(mass)
    Mcm = M_tot/NA
    rho = Mcm/vol
    print("Num = {0:.0f}, M(g)= {1:.2e}, Vol = {2:.2e}, rho = {3:.2f}".format(num,Mcm, vol, rho))
    return num, rho


def H2_List(H_ats):
    lst = []
    for at in H_ats:
        if(np.isin(at,lst)):
            continue
        else:
            for neb in at.neighbors:
                if(neb.type == 1 and atom.distance(at,neb) <= 1.0):
                    lst.append([at,neb])
                else:
                    continue
    return lst

def Four_Coord(at):
    if(len(at.neighbors) >= 4):
        for neb in at.neighbors:
            if(neb.type == 1):
                return False
            else:
                continue
        return True
    else:
        return False

def Create_DB(H2, md):
    out_file = out_path + "/DB-S{0}-H{1:.0f}.dump".format(ARGS[2],H2[0].id)
    center = H2[0].coords + 0.5*atom.Compute_Vec(H2[0], H2[1])
    f_init = [1000,1,0,0,0]
    fake = atom.ATOM(f_init)
    fake.coords = center
    tmp = md.atoms
    at_list = []
    idx = 10000
    min_dist = 4.0
    for at in md.atoms:
        curr_dist = atom.distance(fake,at)
        if(curr_dist < min_dist and Four_Coord(at)):
            min_dist = curr_dist
            idx = at.id
    print("\n###################\n{0:.0f}\n".format(idx))
    md.pop(idx)
    io.Write_Dump(out_file,md)
    md.atoms = tmp
    md.num +=1
    return

def Compute_Ratio(sim):
    #(1) break up the simulation box in to rectangular parallapiped i.e. R = w X w X h for w = 5.43 => 25 regions.
    # what I really want is number of bins x and number of bins y
    num_x = 5
    num_y = 5
    dx = sim.box[0][1]/num_x
    dy = sim.box[1][1]/num_y
    #then loop through atoms checking their x and y values and then assign to a region
    bins = np.zeros([num_x,num_y])
    bin_x = dx/2.0
    bin_y = dy/2.0
    for at in sim.atoms:
        x, y, z = at.coords
        if(z > 16.0 and at.type == 1 ):
            x_idx = math.floor(x/dx)
            y_idx = math.floor(y/dy)
            bins[x_idx][y_idx]+=1
    hist = []
    for i in range(num_x):
        hist_x = i*dx + dx/2.0
        for j in range(num_y):
            hist_y = j*dy + dy/2.0
            hist.append([hist_x, hist_y, bins[i][j]])
    return np.asarray(hist)




def main():
    sim = rl.Read_Data(test_file)
    hist = Compute_Ratio(sim)
    Plot_Hist(hist)

if __name__ =='__main__':
    test_file = "/Users/diggs/Desktop/TOPCon/out-1-9-23/SiOx-1.59.dat"
    ARGS = sys.argv
    FILE=ARGS[1]
    main()
















