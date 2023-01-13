#original file written by Andrew Diggs
import ReadLAMMPS as rl
import atom
import sys, os
import pdb
import numpy as np




def Replicate(sim, x,y,z):
    new_sim = rl.Simulation
    bx, by, bz = sim.box[0][1], sim.box[1][1], sim.box[2][1]
    temp = []
    count = sim.num + 1
    for at in sim.atoms:
        if(at.type == 1):
            at.type =3
        elif(at.type == 2):
            at.type = 1
        ax, ay, az = at.coords
        if((ax + bx) < x):
            temp.append(atom.ATOM([count, at.type, ax + bx, ay, az]))
            count +=1
        if((ay + by) < y):
            temp.append(atom.ATOM([count, at.type, ax, ay + by, az]))
            count +=1
        if((ax + bx) < x and (ay + by) < y):
            temp.append(atom.ATOM([count, at.type, ax + bx, ay + by, az]))
            count +=1
    sim.box = np.array([[0.0,x], [0.0,y], [0.0,bz]])
    atom.ATOM.set_box(sim.box)
    sim.num +=len(temp)
    sim.atoms = np.concatenate([sim.atoms,np.asarray(temp)])
    return





def main():
    test_file = "/Users/diggs/Desktop/LAMMPS/DATA/a-Si-H-12/aSi-H_optimize_8.dat"
    Out_Dir = "/Users/diggs/Desktop/TOPCon/CreateStacks/aSiH/"
    test = rl.Read_Data(test_file)
    print(test.timestep)
    print(test.box)
    print(test.num)
    Replicate(test, 5*5.43, 5*5.43, 0.0)
    print(test.box)
    print(test.num)
    dirs= test_file.split('/')
    Fname = dirs[len(dirs) - 1]
    rl.Write_Dump(Out_Dir + Fname[:-3] + "dump", test)



if __name__ == '__main__':
    main()













