import numpy as np
import linecache as lc
import matplotlib.pyplot as plt
import os
from datetime import datetime
import sys
import re
import pdb
plt.style.use('seaborn-deep')
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['mediumblue', 'crimson','darkgreen', 'darkorange','crimson', 'darkorchid'])
plt.rcParams['figure.figsize'] = [12,8]
plt.rcParams['axes.linewidth'] = 1.7
plt.rcParams['lines.linewidth'] = 6.0
#plt.rcParams['axes.grid'] = True
plt.rcParams['font.size'] = 22
plt.rcParams['font.family'] =  'sans-serif'

#I need to read a reax.dat file and a reax.log file and get values from both.


def read_dat(file):
    my_list = []
    with open(file) as f:
        for line in f.readlines():
            x = line.split()
            if(len(x) == 8):
                vals = [int(x[0]), int(x[1]), float(x[2]),float(x[3]),float(x[4])]
                my_list.append(vals)
    f.close()
    return np.asarray(sorted(my_list, key=lambda x:x[0]))

def read_dump(file, hnum):
    my_list=[]
    with open(file) as f:
        lines=f.readlines()
        box = np.asarray([lines[5].split()[1],lines[6].split()[1],lines[7].split()[1]], dtype = float)
        for line in lines:
            x=line.split()
            if(len(x)==5):
                if(int(x[0]) == int(hnum)):
                    f.close()
                    return np.asarray([int(x[0]), float(x[2])*box[0], float(x[3])*box[1], float(x[4])*box[2]])
    f.close()
    return 


def read_log(file, mod=0):
    ar=[]
    with open(file) as f:
        cont=len(f.readlines())
        line = lc.getline(file,cont - mod).split()
    f.close()
    for val in line:
        ar.append(float(val))
    return np.asarray(ar)
#===============================================================
def MEP(file):
    #there are 9 elements before RD1
    last= read_log(file)
    prev = read_log(file,mod=1)
    R=last[9::2]
    mep=last[10::2]
    ms = mep - np.min(mep)
    efb=last[6]
    erb=last[7]
    RD=last[8]
    return R, ms , efb, erb, RD

def plot_mep(path,file,hnum, xo= 0.01):
    r,pe,EF,ER, RD = MEP(file)
    my_barriers=[]
    points=[]
    mytext="FEB ={0:.3f} REB = {1:.3f}"
    indices, vals, nb = calc_barrier(file)
    if not (nb):
        points.append([0,0,0])
    else:
        for i in range(nb):
            l, p, s = vals[i]
            a,b,c = indices[i]
            feb = p-l
            reb = p-s
            print(mytext.format(feb,reb))
            my_barriers.append([feb,reb])
            points.append([r[a], r[b], r[c]])
    name=file[len(path):-4]
    fig = plt.figure(figsize=[10,7])
    plt.scatter(r,pe, marker = '^', color = 'darkgreen', s=180)
    #plt.plot(r,pe, linestyle = '--', linewidth = 3.0, color = 'darkgreen')
    #plt.scatter(points,vals, color='r', s=20)
    txt=(r"Forward E$_A $ = {0:.2f} eV"+"\n"
        + r"Reverse E$_A $ = {1:.2f} eV"+"\n").format(EF, ER)
    txt2 = "Replica distance\n{0:.2f}".format(RD) + r"$\AA $"
    plt.text(xo, np.max(pe)*0.8, txt, fontsize = 18)
    #plt.text(xo, np.max(pe)*0.68, txt2)
    #plt.title("MEP")
    plt.ylabel("PE (eV)")
    plt.xlabel(r'$x_{replica} $')
    plt.draw()
    plt.waitforbuttonpress(1)
    #print("Do you want to use this barrier?")
    choice = input("Do you want to use original barrier?") # this will wait for indefinite time
    if(choice == "y"):
        write_dat(EF,ER,hnum)
    elif(choice == 'm'):
        c2 = input("which Barriers?")
        for v in c2.split():
            i = int(v)
            feb, reb = my_barriers[i]
            write_dat(feb, reb, hnum)
    elif(choice =='s'):
        plt.savefig(name +".png")
    elif(choice == 'q'):
        exit()
    plt.close(fig)
    return 


def calc_barrier(file):
    r, pe, ef, er, rd = MEP(file)
    low = pe[0]
    emin = 0
    peak = 0
    p_index =0
    m_index = 0
    low_index = 0
    ref =0.0
    climbing = True
    out_points = []
    out_vals =[]
    num_barriers = 0
    #pdb.set_trace()
    for i in range(3,len(pe)):
        if(climbing):
            if(pe[i] > ref):
                ref = pe[i]
                peak = pe[i]
                p_index = i
                continue
            else:
                climbing =False
        if not(climbing):
            #pdb.set_trace()
            if(pe[i] < ref):
                ref = pe[i]
                emin = pe[i]
                if(i == len(pe)-1):
                    out_points.append([low_index,p_index,i])
                    out_vals.append([low,peak, emin])
                    num_barriers+=1
                    continue
                else:
                    continue
            elif(pe[i-2]== peak):
                ref = pe[i]
                if(pe[i]> peak):
                    peak = pe[i]
                    p_index=i
                    climbing = True
                    continue
                else:
                    continue
            else:
                ref = pe[i]
                out_points.append([low_index,p_index,i])
                out_vals.append([low,peak, emin])
                low = emin
                low_index = i
                num_barriers+=1
                climbing = True
                continue
    if(num_barriers):
        return out_points, out_vals, num_barriers
    else:
        return [0,0,0] , [0,0,0] , 0

def get_z(hnum):
    Dname="sample{0}-{1}.dump.{2}"
    dmp1=pth + Dname.format(sys.argv[4],hnum,'01')
    dmp2=pth + Dname.format(sys.argv[4],hnum,'15')
    c1=read_dump(dmp1,hnum)
    c2=read_dump(dmp2,hnum)
    z1, z2= c1[3], c2[3]
    delta=c2 - c1
    dist = np.sqrt(np.sum(delta*delta))
    print(z1,z2)
    print(dist)
    return z1,z2,dist


def neb_dump(first, num_procs,outfile):
    of=open(outfile, 'w')
    ts=re.compile("ITEM: TIMESTEP")
    for i in range(1,num_procs):
        file= (first + "{0:02d}").format(i)
        with open(file) as f:
            lines=f.readlines()
            tot=len(lines)
            itr=0
            for l in lines:
                if(ts.match(l)):
                    itr+=1
                    continue
            block= int(tot/itr)
            start=int(tot - block)
            print(lines[start])
            of.writelines(lines[start:tot])
    of.close()
    return
#==================================================================


def write_dat(feb, reb, hnum):
    ofile=sys.argv[5]
    z1, z2, delta = get_z(hnum)
    line="{0} {1:.4f} {2:.4f}\n".format(int(hnum), feb, reb)
    outfile = open( ofile, "a")
    outfile.write(line)
    outfile.close()
    return 


def get_num(path, file):
    p1 = len(path)
    prefix="S{0}-H".format(sys.argv[4])
    p2=len(prefix)
    offset=p1+p2
    return int(file[offset:-4])


if __name__=='__main__':
    pth=sys.argv[1]
    hnum=get_num(pth,sys.argv[2])
    plot_mep(pth, sys.argv[2],hnum)


















