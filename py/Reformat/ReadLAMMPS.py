#original file written by Andrew Diggs
import sys
import os
import re
import linecache as lc
import atom
import numpy as np
import multiprocessing as mp
import pdb
import time
#====This is a file that will be good at reading lammps and QE files=============
# ====Steps to making this work well are as follows=========
# 1) get file name and path========
ITEMS=re.compile('ITEM:')
TIMESTEP=re.compile('ITEM: TIMESTEP')
NUM=re.compile('ITEM: NUMBER OF ATOMS')
BB=re.compile('ITEM: BOX BOUNDS')
ATOMS=re.compile('ITEM: ATOMS \w')

DatNum = re.compile(r'\d+\s+atoms')
DatTypes = re.compile(r'\d\s+atom types')
DatBox = re.compile(r'\d+\s+\d+\.\d+\s+[x-z]lo [x-z]hi')
DatBoxx = re.compile(r'\d+\.\d+\s+\d+\.\d+\s+xlo xhi')
DatBoxy = re.compile(r'\d+\.\d+\s+\d+\.\d+\s+ylo yhi')
DatBoxz = re.compile(r'\d+\.\d+\s+\d+\.\d+\s+zlo zhi')
DatAtomCoords = re.compile(r'\s+?\d+\s+\W')
DatQCoords = re.compile(r'\s*\d+\s\d\s-?\d')
DatAtoms = re.compile(r'Atoms\s+\#?\s+?\w?')
DatVelocities = re.compile(r'Velocities')






s_timestep = "ITEM: TIMESTEP\n{0:.0f}\n"
s_num = "ITEM: NUMBER OF ATOMS\n{0:.0f}\n"
s_bb= "ITEM: BOX BOUNDS pp pp pp\n{0:.6e} {1:.6e}\n{2:.6e} {3:.6e}\n{4:.6e} {5:.6e}\n"
s_atoms = "ITEM: ATOMS id type xs ys zs\n"

Fstart = 0
end = 500

Box_Dict = {'xlo': None, 'xhi': None, 'ylo': None, 'yhi': None, 'zlo': None, 'zhi': None, 'count': 0}

class Simulation():
    
    def __init__(self):
        self.timestep = None
        self.num = None
        self.box = np.empty([3,2])
        self.atoms = None
        self.done = False



    def set_Timestep(self,line):
        val = line.split()
        self.timestep=int(val[0])

    def set_Num(self,line):
        val = line.split()
        self.num=int(val[0])
        atom.ATOM.set_num(self.num)
        self.atoms = np.empty([self.num],dtype=atom.ATOM)

    def set_BB_Dump(self,lines):
        for j in range(3):
            l=lines[j].split()
            self.box[j]=[float(l[0]),float(l[1])]
        atom.ATOM.set_box(self.box)

    def set_Atoms(self,lines,scaled = True):
        #pdb.set_trace()
        tmp=np.empty([self.num],dtype=atom.ATOM)
        if(scaled):
            s=np.diff(self.box,axis=1).reshape(3)
        else:
            s= [1.0,1.0,1.0]
        print(s)
        for line in lines:
           l=line.split()
           x=[int(l[0]),int(l[1]),float(l[2])*s[0],float(l[3])*s[1],float(l[4])*s[2]]
           tmp[x[0]- 1]=atom.ATOM(x)
        self.atoms=tmp
        self.done=True
        return

    def set_BB_Dat(self, line):
        v = line.split()
        Box_Dict[v[2]] = float(v[0])
        Box_Dict[v[3]] = float(v[1])
        Box_Dict['count']+=1
        if(Box_Dict['count'] == 3):
            self.box[0][0] = Box_Dict['xlo']
            self.box[0][1] = Box_Dict['xhi']
            self.box[1][0] = Box_Dict['ylo']
            self.box[1][1] = Box_Dict['yhi']
            self.box[2][0] = Box_Dict['zlo']
            self.box[2][1] = Box_Dict['zhi']
            atom.ATOM.set_box(self.box)
        return

    def set_Atom(self,line, charge = False):
        s = [1.0, 1.0, 1.0]
        l=line.split()
        if(charge):
            x=[int(l[0]),int(l[1]),float(l[3])*s[0],float(l[4])*s[1],float(l[5])*s[2]]
        else:
            x=[int(l[0]),int(l[1]),float(l[2])*s[0],float(l[3])*s[1],float(l[4])*s[2]]
        self.atoms[x[0]- 1]=atom.ATOM(x)


    def Set_Neighbors(self):
        start = time.time()
        for at in self.atoms:
            at.get_neighs(self.atoms)
        end = time.time()
        print("Wall Time: {0:.3f}".format(end - start))
        return 1

    def pop(self, num):
        mask = np.ones(self.num, dtype = bool)
        mask[num - 1] = False
        self.atoms[self.num - 1].id = num
        #self.atoms[num - 1] = self.atoms[self.num - 1]
        self.atoms = self.atoms[mask]
        self.num -= 1
        print(self.num)
        return 

# I want to read a file
# iterate through each lines
# if ^lines == ITEMS:
# read the next CAPS WORD/S 
# go to switch statment
# if next WORD == TIMESTEP; set timestep
# and so on...
def Read_Dump(file):
    inst=Simulation()
    f=open(file, 'r')
    lines=f.readlines()
    f.close()
    for i in range(len(lines)):
        if(inst.done):
            if(ITEMS.match(lines[i])):
                break
            else:
                continue
        elif(ITEMS.match(lines[i])):
            set_vals(inst,lines,i)
    inst.Set_Neighbors()
    return inst


def Read_Data(infile):
    inst = Simulation()
    inst.timestep = 0
    with open(infile) as f:
        lns = f.readlines()
        for l in lns:
            if(DatVelocities.match(l)):
                break
            elif(DatNum.match(l)):
                inst.set_Num(l)
            elif(DatBox.match(l)):
                inst.set_BB_Dat(l)
            elif(DatQCoords.match(l)):
                inst.set_Atom(l,charge=True)
            else:
                continue
    return inst

def set_vals(inst,lines,i):
    if(NUM.match(lines[i])):
        #print(lines[i])
        inst.set_Num(lines, i)
    elif(BB.match(lines[i])):
        #print(lines[i])
        inst.set_BB_Dump(lines, i)
    elif(ATOMS.match(lines[i])):
        #print(lines[i])
        lst=lines[i+1:i+1+inst.num]
        inst.set_Atom(lst)
    elif(TIMESTEP.match(lines[i])):
        print(lines[i])
        inst.set_Timestep(lines[i], i)
    else:
        print("????")
        print(lines)


### the current issue is reading in multiple time stemps 



def Write_Dump(out_file,Simulation):
    at_line="{0} {1:.0f} {2:.6f} {3:.6f} {4:.6f}\n"
    out = open(out_file, 'w')
    out.write(s_timestep.format(Simulation.timestep))
    out.write(s_num.format(Simulation.num))
    out.write(s_bb.format(*Simulation.box.flatten()))
    out.write(s_atoms)
    for at in Simulation.atoms:
        xs, ys, zs = at.scaled_coords()
        out.write(at_line.format(at.id,at.type,xs,ys,zs))
    return



def Write_NEB(moved,ID,infile):
    s=np.diff(atom.ATOM.box,axis=1).reshape(3)
    ids=[]
    for at in moved:
        ids.append(at.id)
    ids=np.asarray(ids)
    at_line="{0} {1} {2:.6f} {3:.6f} {4:.6f}\n"
    ofile='outfiles/final_H_{0}.dat'.format(ID)
    if(len(moved) ==1):
        check = moved[0].distance_moved()
        if(check <= 0.1):
            return None
    
    off=open(ofile, 'w')
    with open(infile) as f:
        for l in f.readlines():
            x=l.split()
            if(ITEMS.match(l)):
                off.write(l)
                continue
            elif(len(x) != 5):
                off.write(l)
                continue
            else:
                rep=np.isin(ids,int(x[0]))
                if(np.any(rep)):
                    at=moved[rep][0]
                    tmp=at.coords/s
                    fline=at_line.format(at.id, at.type, tmp[0], tmp[1], tmp[2])
                    off.write(fline)
                    at.reset()
                    continue
                else:
                    off.write(l)
                    continue
    off.close()
    return ofile

def File_Comp(initial,final):
    print("#####FILE COMP#######")
    s=open(initial, 'r').readlines()
    f=open(final, 'r').readlines()
    for i in range(len(f)):
        if(s[i] != f[i]):
            print(s[i])
            print(f[i])
    print("")
    return 2


def Write_Density(file,sample, num_tot,rho, num_H2):
    txt = "{0} {1:.0f} {2:.2f} {3:.0f}\n"
    out = open(file, 'a')
    out.write(txt.format(sample,num_tot, rho, num_H2))
    out.close()
    return

def Write_H2(file,sample,H2):
    #pdb.set_trace()
    txt = "{0} H[{1:.0f}]-H[{2:.0f}]\n"
    out = open(file, 'a')
    for hh in H2:
        out.write(txt.format(sample,hh[0].id, hh[1].id))
        print(txt.format(sample,hh[0].id, hh[1].id))
    out.close()
    return





#=========this section contains read an write for .dat files
#d_atoms = re.compile(r'[[:digit:]]( atoms)')





def Write_Data(in_file,Ens):
    Out = open(in_file[:-4]+"dat", 'w')
    Head = "LAMMPS data file via write_data, version 29 Sep 2021, timestep = {0:.0f}\n".format(Ens.timestep)
    Num = "{0:.0f} atoms\n".format(Ens.num)
    Typs = "2 atom types\n"
    Atoms = "{0:.0f} {1:.0f} {2:.6f} {3:.6f} {4:.6f} 0 0 0\n"
    Vel = "{0:.0f} 0 0 0\n"
    Out.write(Head)
    Out.write("\n")
    Out.write(Num)
    Out.write(Typs)
    Out.write("\n")
    Box = "{0:.5f} {1:.5f} {2}lo {3}hi\n"
    Axis = ['x','y', 'z']
    for i in range(3):
        Out.write(Box.format(Ens.box[i][0],Ens.box[i][1],Axis[i],Axis[i]))
    Out.write("\n")
    Out.write("Masses\n")
    Out.write("\n")
    Out.write("1 1.008\n2 28.0855\n")
    Out.write("\n")
    Out.write("Atoms # atomic\n")
    Out.write("\n")
    for at in Ens.atoms:
        Out.write(Atoms.format(at.id, at.type, *at.coords))
    Out.write("\n")
    Out.write("Velocities\n")
    Out.write("\n")
    for at in Ens.atoms:
        Out.write(Vel.format(at.id))
    Out.close()
    return

def Write_Data_From_Dump(in_file, start):
    Ens = setup_fast(in_file, start)
    Out = open(in_file[:-3]+"dat", 'w')
    Head = "LAMMPS data file via write_data, version 29 Sep 2021, timestep = {0:.0f}\n".format(Ens.timestep)
    Num = "{0:.0f} atoms\n".format(Ens.num)
    Typs = "2 atom types\n"
    Atoms = "{0:.0f} {1:.0f} {2:.6f} {3:.6f} {4:.6f} 0 0 0\n"
    Vel = "{0:.0f} 0 0 0\n"
    Out.write(Head)
    Out.write("\n")
    Out.write(Num)
    Out.write(Typs)
    Out.write("\n")
    Box = "{0:.5f} {1:.5f} {2}lo {3}hi\n"
    Axis = ['x','y', 'z']
    for i in range(3):
        Out.write(Box.format(Ens.box[i][0],Ens.box[i][1],Axis[i],Axis[i]))
    Out.write("\n")
    Out.write("Masses\n")
    Out.write("\n")
    Out.write("1 1.008\n2 28.0855\n")
    Out.write("\n")
    Out.write("Atoms # atomic\n")
    Out.write("\n")
    for at in Ens.atoms:
        Out.write(Atoms.format(at.id, at.type, *at.coords))
    Out.write("\n")
    Out.write("Velocities\n")
    Out.write("\n")
    for at in Ens.atoms:
        Out.write(Vel.format(at.id))
    Out.close()
    return



def test_regex(file, reg):
    with open(file) as f:
        lns = f.readlines()
        for l in lns:
            if(reg.match(l)):
                print(l)


if __name__ == "__main__":
    DatBox = re.compile(r'\d+\s+\d+\.\d+\s+[x-z]lo [x-z]hi')
    test_file = "/home/agoga/topcon/output-farm/AnnealLoopSiOx-law50h-60429703-S/SlowishAnnealDT75.dump"
    Read_Dump(test_file)
#DatNum = re.compile(r'\d+\s+atoms')
#DatTypes = re.compile(r'\d\s+atom types')
#DatBoxx = re.compile(r'\d+\.\d+\s+\d+\.\d+\s+xlo xhi')
#DatBoxy = re.compile(r'\d+\.\d+\s+\d+\.\d+\s+ylo yhi')
#DatBoxz = re.compile(r'\d+\.\d+\s+\d+\.\d+\s+zlo zhi')
#DatAtomCoords = re.compile(r'\s+?\d+\s+\W')
#DatAtoms = re.compile(r'Atoms\s+\#?\s+?\w?')
#DatVelocities = re.compile(r'Velocities')














