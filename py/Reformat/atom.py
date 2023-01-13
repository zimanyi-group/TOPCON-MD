#original file written by Andrew Diggs
import numpy as np


#==============
#boundary wrapper 
# only use if atom.x + cuttoff goes through the simbox
#cutoff=2.8 #angstrom
# when we are looking for neighbors we will measure the distance from atom-a to atom-b
# if the distance is greater than 1/2 the sim box use wrapped coords
#step 1 measure distance()
def Compute_Vec(a1, a2):
    vec=np.empty([3],dtype = float)
    if(a1.id == a2.id):
        x1, x2 = a1.coords, a1.initial
    else:
        x1, x2 = a1.coords, a2.coords
    dist= 0
    for i in range(3):
        box_length=ATOM.box[i][1] - ATOM.box[i][0]
        delta=abs(x1[i] - x2[i])
        if(delta > 0.5*box_length):
            delta = least_dist([x1[i],x2[i]],box_length)
        vec[i] = delta
    return vec


def distance(a1,a2):
    if(a1.id == a2.id):
        x1, x2 = a1.coords, a1.initial
    else:
        x1, x2 = a1.coords, a2.coords
    dist= 0
    for i in range(3):
        box_length=ATOM.box[i][1] - ATOM.box[i][0]
        delta=abs(x1[i] - x2[i])
        if(delta > 0.5*box_length):
            delta = least_dist([x1[i],x2[i]],box_length)
        dsquare = delta**2
        dist+=dsquare
    return np.sqrt(dist)
    
def least_dist(s,bl):
    x1, x2 = s
    if(x1>x2):
        return abs((x2 + bl) - x1)
    else:
        return abs((x1 + bl) - x2)

def boundary_wrapper(ar):
    new_coords=np.empty([3])
    for i in [0,1,2]:
        lb, ub =ATOM.box[i]
        if(ar[i] < lb):
            new_coords[i] = ar[i] + ub
        elif(ar[i] > ub):
            new_coords[i] = ar[i] - ub
        else:
            new_coords[i] = ar[i]
    return new_coords

def Compute_Volume(cutoff):
    box_length = []
    x = ATOM.box[0][1] - ATOM.box[0][0]
    y = ATOM.box[1][1] - ATOM.box[1][0]
    z = ATOM.box[2][1] - cutoff
    # there are 10^7 A in one cm => 1 = 1cm/10^7A
    return (x*y*z)*1.0e-24
#===========================================
class ATOM():
    cutoff=2.8
    box=None
    NUM = None
    
    def __init__(self, vals):
        self.id = vals[0]
        self.type = vals[1]
        self.x = vals[2]
        self.y = vals[3]
        self.z = vals[4]
        self.neighbors = []
        self.initial=np.array([vals[2], vals[3], vals[4]])
    
    @property
    def coords(self):
        return np.array([self.x, self.y, self.z])
    
    @coords.setter
    def coords(self, ar):
        self.x, self.y, self.z = boundary_wrapper(ar)

    def scaled_coords(self):
        s=np.diff(ATOM.box,axis=1).reshape(3)
        return np.divide(self.coords,s)

    def get_neighs(self,array_of_atoms):
        for i in range(self.id,ATOM.NUM):
            at = array_of_atoms[i]
            if(distance(self,at) <= ATOM.cutoff):
                self.neighbors.append(at)
                at.neighbors.append(self)
        self.neighbors=np.asarray(self.neighbors)

    def reset(self):
        self.coords = self.initial
    
    def distance_moved(self):
        atype=['oops!', 'H', 'Si']
        d=distance(self,self)
        txt="{0} atom {1} moved {2:.3f}".format(atype[self.type], self.id, d)
        print(txt)
        return d
    
    @classmethod
    def set_box(cls,sim_box):
        cls.box=sim_box
        return

    @classmethod
    def set_num(cls,num):
        cls.NUM=num
        return
#======================================================================
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        