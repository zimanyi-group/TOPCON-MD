import numpy as np
import linecache as lc
from mpi4py import MPI
import matplotlib.pyplot as plt
import os
from datetime import datetime
import sys
from matplotlib.ticker import MaxNLocator
import csv 
# import itertools
import collections

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
me = MPI.COMM_WORLD.Get_rank()

if me !=0:
    quit()
    
    
plt.style.use('seaborn-deep')
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['mediumblue', 'crimson','darkgreen', 'darkorange','crimson', 'darkorchid'])
#plt.rcParams['figure.figsize'] = [12,8]
plt.rcParams['axes.linewidth'] = 1.7
plt.rcParams['lines.linewidth'] = 6.0
#plt.rcParams['axes.grid'] = True
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] =  'sans-serif'
plt.rcParams["figure.autolayout"] = True

#conversion from kcal/mol to eV
conv=0.043361254529175
#I need to read a reax.dat file and a reax.log file and get values from both.

etol=sys.argv[3]
timestep=sys.argv[4]


def read_log(file, mod=0):
    ar=[]
    with open(file) as f:
        cont=len(f.readlines())
        line = lc.getline(file,cont - mod).split()
    f.close()
    for val in line:
        ar.append(float(val))
    return np.asarray(ar)


def MEP(file):
    #there are 9 elements before RD1
    last= read_log(file)
    prev = read_log(file,mod=1)
    R=last[9::2]                #points in replica space, RD1-RDN
    mep=last[10::2]*conv            #actual pE values, PE1-PEN
    ms = mep - np.min(mep)      #normalizing MEP
    efb=last[6]*conv               #forward barrier
    erb=last[7]*conv               #reverse barrier
    RD=last[8]                  #total reaction cood space
    return R, ms , efb, erb, RD

def plot_mep(path,file,fileID,hnum=0, xo= 0.01):
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
            
    name=fileID
    
    fig = plt.figure(figsize=[6,6])
    plt.scatter(r,pe, marker = '^', color = 'darkgreen', s=180)
    #plt.plot(r,pe, linestyle = '--', linewidth = 3.0, color = 'darkgreen')
    #plt.scatter(points,vals, color='r', s=20)
    txt=(r"Forward E$_A $ = {0:.2f} eV"+"\n"
        + r"Reverse E$_A $ = {1:.2f} eV"+"\n").format(EF, ER)
    txt2 = "Replica distance={0:.2f}".format(RD) + r"$\AA $"
    plt.text(xo, np.max(pe)*0.8, txt, fontsize = 14)
    
    plt.text(xo, np.max(pe)*0.68, txt2,fontsize=14)
    plt.title(f"MEP with E-tol': {etol} & timestep: {timestep}")
    plt.ylabel("PE (eV)")
    plt.xlabel(r'$x_{replica} $')
    #plt.axes().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid('on',axis='y',linewidth=1)
    plt.savefig(path+name +"-NEB.png")
    
    return (EF,ER,my_barriers,RD)


def calc_barrier(file):
    r, pe, ef, er, rd = MEP(file)
    pe=pe
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


def savecsv(data,filename,col_names=None):

    csv_name=filename+'.csv'


    file_exists = os.path.isfile(filename)

    #data=runname+','+data+''
    with open(filename,'a',newline='', encoding='utf-8') as fd:
        csv_writer=csv.writer(fd)

        if file_exists is False and col_names is not None:
            csv_writer.writerow(col_names)
            
        csv_writer.writerow(data)

def catch(func, *args, handle=lambda e : e, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        #print(handle(e))
        return None

def calc_dist():
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import pandas
    import numpy as np
    import statistics

    base="/home/agoga/documents/code/topcon-md/output/NEB/"#1000pairs-v2/"



    d=base+"pairs.csv"
    data = pandas.read_csv(d)


    dist=[]
    done=[]
    bad=[]#["78-90","78-89","168-178","171-899","258-270","258-363","265-270"]
    total=0
    skip=0
    for index,row in data.iterrows():

        sum=0
        over=0
        pair = row['pair']
        if pair not in done and pair not in bad:
            for i,r in data[data.pair==pair].iterrows():
                if r['dist'] < 3.6:
                    feb = r['A']
                    if feb < 6:
                        sum += feb
                        over+=1

            if over >0:
                avg = sum/over
                dist.append(avg)
            done.append(pair)
            total+=1
        else:
            print(pair)
            skip+=1
            
            
    print(len(dist))
    #counts,bins=np.histogram(dist)
    print(total)
    print(skip)

    mean=statistics.mean(dist)
    stddev=statistics.stdev(dist)
    mstxt=f"Mean: {round(mean,2)}\nStd Dev: {round(stddev,2)}"

    f,ax = plt.subplots()
    plt.title(f'NEB Barrier Distribution for {total} runs')
    plt.xlabel('FEB(eV)')
    plt.ylabel('Counts')
    plt.hist(dist,18)

    plt.text(0.01,0.99,mstxt,ha='left',va='top', transform=ax.transAxes, fontsize = 10)
    plt.show()




#example sequence
    #3000    57.299944    1038.8458.......
    #Climbing replica = 6
    #Step MaxReplicaForce MaxAtomForce.....
    #3000    57.295123........
#end example
def check_convergence(filename,maxneb,maxclimbing,numpart=13):
    
    fsize=os.path.getsize(filename)
    with open(filename,'r') as logF:
        last = collections.deque(maxlen=1)
        climbing=None
        nebiter=None
        climbing_iter_i=None
        climbing_iter_f=None
        for line in logF:
            fsize -= len(line)
            
            if not fsize:#this line is the last line
                climbing_iter_f=int(line.split()[0])
                
            elif line.startswith('Climbing'):
                nebiter=int(last[0].split()[0])
                climbing_iter_i=nebiter#if god loves us this is easy else use next=itertools.islice(logF,2)
                climbing=int(line.split()[3])
            
            last.append(line)
        print('got her')
        if climbing is None or nebiter is None or climbing_iter_f is None or climbing_iter_i is None:#cry
            print('Big boy oppsie in "Process-NEB.py"')    
            return False
        elif climbing==numpart or climbing==0:
            print('NEB did not converge because the climbing replica was the first or last replica.')  
            return False
        elif nebiter == maxneb:
            print('NEB did not converge because the maximum NEB iterations were reached.')  
            return False
        elif (climbing_iter_f-climbing_iter_i) == maxclimbing:
            print('NEB did not converge because the maximum climbing image iterations were reached.')  
            return False
        else:
            return True
    

if __name__=='__main__':
    
    dirname=sys.argv[1]
    atomID=sys.argv[2]
    removeID=sys.argv[5]
    nebfolder=sys.argv[6] #second+"/NEB/"
    datafile=sys.argv[7]
    
    
    fileID=atomID
    
    csvID=str(atomID)+'-'+str(removeID)
    col_names=["pair","etol","ts","dist","FEB","REB","A","B","C","D","E","F","G","H"]
    
    #dirname="/home/agoga/documents/code/topcon-md/data/HNEB1/"#os.path.dirname(os.path.realpath(pth))
    
    
    
    file=f"{dirname}logs/{fileID}neb.log"

    splt=dirname[:-1].split("/")
    tname=splt[-1]#name of the output folder 'NEB125-126_1-0.01_11'
    second="/".join(splt[:-1])
    
    datend=".data"
    if datafile.endswith(datend):
        datafile=datafile[:-len(datend)]
    
    
    #print(f"splt: {splt}, tname: {tname}, second: {second}, nebfolder: {nebfolder}")
    csvfile=nebfolder+datafile+".csv"
    
    if check_convergence(file,3000,1000):
        
    
        ret=plot_mep(dirname,file,fileID)#,hnum)
        
        
        import numpy as np
        from PIL import Image

        list_im = [dirname+f"{fileID}-NEB.png",dirname+f"PES({fileID}).png",dirname+f"{fileID}-Ovito.png"]
        
        imgs=[]
        for i in list_im:
            im=catch(Image.open,i)
            if im is not None:
                imgs.append(im)
                
                
        # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
        min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
        imgs_comb = np.hstack([i.resize(min_shape) for i in imgs])

        # save that beautiful picture
        imgs_comb = Image.fromarray(imgs_comb)
        imgs_comb.save(dirname+f"Full.png")    
        imgs_comb.save(nebfolder+tname[3:] +".png")
        
        
        

        
        dat=[csvID,etol,timestep,ret[3],ret[0],ret[1]]
        
        for l in ret[2]:
            dat.append(l[0])
            dat.append(l[1])
            
        savecsv(dat,csvfile,col_names)
    else:

        # dat=[csvID,etol,timestep]
    
            
        # savecsv(dat,csvfile,col_names)
        print('No convergence, no cry')
    # for p in list_im:
    #     try:
    #         os.remove(p)
    #     except:
    #         i=0



