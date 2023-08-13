from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np
import statistics
from scipy import stats
from pathlib import Path
from lammps import lammps
import dask.dataframe as dd
from dask.multiprocessing import get
from operator import itemgetter
from math import degrees
import pandas as pd
pd.options.mode.chained_assignment = None 
from ast import literal_eval
import os
from mpi4py import MPI
import time
import matplotlib.cm as cm
import matplotlib.colors as colors
from ase.geometry import get_angles
import ase.cell

from matplotlib.ticker import MaxNLocator
from pandas.api.types import is_numeric_dtype
jp=0


me = MPI.COMM_WORLD.Get_rank()
numproc=MPI.COMM_WORLD.Get_size()



SiOBondOrder=0.9
image_folder="/home/agoga/documents/code/topcon-md/neb-out/analysis-images/"
datafolder="/home/agoga/documents/code/topcon-md/data/neb/"

v=4.74e-20
w=1.6e-7
HNumToConcentration=w/v

def pbc_dist(simbox, pos1,pos2):
    total = 0
    for i, (a, b) in enumerate(zip(pos1, pos2)):
        delta = abs(b - a)
        dimension=simbox[i,1]-simbox[i,0]
        if delta > dimension- delta:
            delta = dimension - delta
        total += delta ** 2
    return total ** 0.5
           
    


def apply_dist_from_pos(df,simbox,pos,atomtype=None):
    distdf=df
    #pos=df.at[atom,'pos']
    
    if atomtype is not None:
        distdf=distdf[distdf["type"]==atomtype]
    #partdf=dd.from_pandas(distdf,npartitions=numproc)
    # print(distdf.iloc[0].to_string())
    #.swifter.progress_bar(False).allow_dask_on_strings(enable=True).apply
    distdf['dist']=distdf.apply(lambda row: pbc_dist(simbox,row['pos'],pos),axis=1)
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

def angle_between_pts(p1, p2,pm,box,debug=False):
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
    
    ang=get_angles([v1],[v2],cell,True)

    print(f"--- post fixing ---\np1{p1} p2{p2}\nimiddle{imiddle} fmiddle{fmiddle}\nbox{box}\nang={ang}") if debug else None
    return (ang,[v1,v2])

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
        
        pair_style	    reaxff /home/agoga/documents/code/topcon-md/potential/topcon.control
        pair_coeff	    * * /home/agoga/documents/code/topcon-md/potential/ffield_Nayir_SiO_2019.reax Si O H

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

#returns (df, simbox)
def read_file_data_bonds(datapath,dfile):
    
    (dfdata,simbox)=read_data(datapath+dfile)
    filename=dfile.removesuffix('.dat').removesuffix('.data').removesuffix('.dump')
    bondfile=filename+".bonds"
    create_bond_file(datapath,dfile,bondfile)
    df=read_bonds(dfdata,datapath+'/scratchfolder/'+bondfile)

    return (df,simbox)



from scipy.sparse import csr_matrix
import networkx as nx

def sparse_graph(df):     
    csvlist=df["csvname"].unique()

    for csvfile in csvlist:

        print(csvfile)


        dfc=df[df["csvname"]==csvfile]

        (bonddf,box)=load_data_and_bonds_from_csv(csvfile)
        
        numatoms=len(bonddf)
        
        sg=nx.Graph()

        for i in range(1,numatoms+1):
            bonds=bonddf.at[i,"bonds"]
            bl=[]
            sg.add_node(i)
            for b in bonds:
                if b[0] != i:
                    sg.add_edge(i,b[0])
        
        
        sg.add_edge(1,2)
        sg.add_edge(2,3)
        sg.add_edge(1,3)
        # for s in sg.edges:
        #     print(s)
        
        #print(list(nx.edge_dfs(sg,5)))
        
        tstart=time.time()
        # for s in sg.nodes:
        ti=time.time()
        cycle=nx.find_cycle(sg,10,orientation='reverse')
        # for start_node in sg.nbunch_iter(s):
        #     print(start_node)
        l=list(cycle)
        tf=time.time()
        print(f"){l} took {tf-ti}s")
            
            


        return
            
def get_path(Pr, i, j):
    path = [j]
    k = j
    while Pr[i, k] != -9999:
        path.append(Pr[i, k])
        k = Pr[i, k]
    return path[::-1]


def find_rings(g,ring_size):

    D,Pr =csg.shortest_path(g,directed=False, return_predecessors=True)
    
    
    for r in range(1,4):
        inds=list(zip(*np.where(D==r)))

        for i in inds:
            print(get_path(Pr,i[0],i[1]))
        
        

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
    for csvpath in csvlist:
        df = pd.read_csv(csvpath)
        csvn=csvpath.split('/')[-1]
        df["csvname"]=""
        df=df.assign(csvname=csvn)
        dflist.append(df)


    combodf=pd.concat(dflist,ignore_index=True)


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


def angle_between_pts_df(df,col,valcol):
    #df=basedf.copy()
    csvlist=df["csvname"].unique()
    
    for csvfile in csvlist:
        dfc=df[df["csvname"]==csvfile]
        (ratio,Hnum)=stats_from_csv_name(csvfile)
        
 
        (posdf,box)=load_data_and_bonds_from_csv(csvfile)
        
        for index, row in dfc.iterrows():
            curcol=row[col]
            pair=row['pair']
            ipos=row['iPos']
            fpos=row['fPos']
            box=np.array(row['box'])
            mover=int(pair.split('-')[0])
            zapped=int(pair.split('-')[1])
            fsn=find_movers_neighbor(posdf,mover,zapped)

            middle=posdf.at[fsn,"pos"]
            
            debug = False
            # if pair=="4704-5380":
            #     debug = True
                
            (ang, ret) = angle_between_pts(ipos,fpos,middle,box,debug)
            
            # if ang <20:
            #     print(f"fudge and crackers {pair}")
            #     print(f"box:{box}\ni:{row['iPos']} f:{row['fPos']}\n m: {middle}\n -- v1:{ret[0]} v2:{ret[1]}")

            df.at[index,valcol]=ang[0]
        
        print(f'angle_between_pts_df done with {csvfile}')
    return df


def plot_pair_angle(basedf,col="FEB"):
    df=basedf.copy()
    csvlist=df["csvname"].unique()
    subfolder="OSiO-bond-angle/"
    colranges=[0,1.8,2.8,3.8,10]
    valcol="Pair angle"
    
    df=angle_between_pts_df(df,col,valcol)
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
        
        
        ax1.scatter(al,fl,s=6)#,facecolors='none',linewidths=0.4)
        ax1.set_title(title)
        ax1.set_xlabel(f"O-Si-O pair angle({units})")
        ax1.set_ylabel(f"{col}(eV)")
        # get legend handles and their corresponding labels
    avgSiO2=109.4
    plt.axvline(avgSiO2,linestyle="dashed")
    plt.text(110,.8,f'Average O-Si-O bond angle in a-SiO2:{avgSiO2}{units}', size=9)
    # handles, labels = ax1.get_legend_handles_labels()

    # # zip labels as keys and handles as values into a dictionary, ...
    # # so only unique labels would be stored 
    # dict_of_labels = dict(zip(labels, handles))

    # # use unique labels (dict_of_labels.keys()) to generate your legend
    # ax1.legend(dict_of_labels.values(), dict_of_labels.keys())
          
    plt.show()
    
    
    
#Plotters
def plot_multi_distribution(setdf,col="FEB",units="eV",plot=False):
    df=dist_from_df(setdf,col,units,plot)
    
    
    #1%, 3%, 5%, 6%, 7%, 9%, 11%, 13%

    fig, (ax1,ax2)=plt.subplots(2,1,sharex=True,height_ratios=[15,1])

    fig.subplots_adjust(hspace=0.1) 
    t=0

    ratios=df["ratio"].unique()
    
    for r in ratios:


        dfr=df[df["ratio"]==r]

        dfr=dfr.sort_values(by=['Hnum'])

        xvals=[]#np.array([0,1,3,5,6,7,9,11,13])

        min=10
        max=0
        
        xvals=dfr['Hnum'].to_numpy(dtype=float)
        yvals=dfr['mean'].to_numpy(dtype=float)
        xvals=xvals*HNumToConcentration

        #Setting tick and label text to smaller font
        axisfontsize=10
        fig.suptitle(f'{col} distributions for Oxygen migration in SiOx',fontsize=axisfontsize+1)

 
        ax2.tick_params(axis='both', which='major', labelsize=axisfontsize)
        ax2.tick_params(axis='both', which='minor', labelsize=axisfontsize)
        ax1.tick_params(axis='both', which='major', labelsize=axisfontsize)
        ax1.tick_params(axis='both', which='minor', labelsize=axisfontsize)
        
        

        fig.supxlabel('Areal Hydrogen Density(H/cm$^2$)',fontsize=axisfontsize)
        fig.supylabel(f'{col} - mean ({units})',fontsize=axisfontsize)

        ax2.set_yticks((0,10))
        ax2.set_ylim(0,0.5)
        # ax1.set_ylim(min-.1,max+.1)
    

        ax1.spines.bottom.set_visible(False)
        ax2.spines.top.set_visible(False)
        ax1.xaxis.tick_top()
        ax1.set_xscale('log')



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


        erb=ax1.plot(xvals, yvals, label=str(r))#,yerr=yer[1])
        ax1.legend(title="Value of x")
        t+=1
    
    avgSiO2=109.4
    ax1.axhline(avgSiO2,linestyle="dashed")
    ax1.text(0.3,.08,f'Average O-Si-O bond angle in a-SiO2:{avgSiO2}{units}', size=9,transform=ax1.transAxes)
    
    name=f"{col}vH"
    dirname=image_folder
    os.makedirs(dirname, exist_ok=True)
    figname=dirname+name+'.svg'
    print(f"Saved figure with path {figname}")
    fig.savefig(figname)
    plt.close(fig)
        #erb[-1][0].set_linestyle('--')
     

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
        ax.legend(dict_of_labels.values(), dict_of_labels.keys())
        ax.set_ylim(minv-limbuf,maxv+limbuf)
        ax.set_title(f"Avg {vartitle} between {alow} eV and {ahigh} eV")
        ax.set_xlabel("H Num")
        ax.set_ylabel(f"Avg {vartitle}")
    plt.show()

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
        print(csvfile)
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
        
    units=r"$$\AA$$"
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
    vartitle="angle(°)"
    plotfilename="angle-histogram"
    print("Calculation done, now plotting")
    plot_any_split_hist(dfa,col,var,colranges,numbins,vartitle,xlabel,units,subfolder,plotfilename)
 

 
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
    subfolder="angle-histogram/"
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
    cols=["ratio","Hnum","mean","stddev"]
    for c in cols:
        setdf[c]=""
        
    setdf.set_index("csvname", inplace=True)
    for csvfile in csvfiles:
        csdf=grpdf[grpdf["csvname"]==csvfile]

        print(csvfile)
        dist=csdf[col].tolist()
        
        (ratio,Hnum)=stats_from_csv_name(csvfile)
        setdf.at[csvfile,"ratio"]=ratio
        setdf.at[csvfile,"Hnum"]=Hnum
        setdf.at[csvfile,"mean"]=mean=statistics.mean(dist)
        setdf.at[csvfile,"stddev"]=stddev=statistics.stdev(dist)
        
        
        setdf[cols]=setdf[cols].apply(pd.to_numeric, errors='coerce')
        mstxt=f"Mean: {round(mean,2)}\nStd Dev: {round(stddev,2)}"
        total=len(csdf)

        
        num_bins=50
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

if __name__ == "__main__":       
        
    nebfolder="/home/agoga/documents/code/topcon-md/neb-out/"


    group=["FullSet/"] #"farm/6169019/","farm/6169020/","farm/6169021/","farm/6169022/"]#
    #group=["farm/perpPairs/","farm/parallelPairs/"]

    multiplot=[]


    for g in group:
        csvlist=[]
        gpath=nebfolder+g
        csvlist=[]
        for d in Path(gpath).glob('*.csv'):
            csvlist.append(str(d))
        
        dstats=[]
        #setdf=csv_to_df(csvlist[0])
        #df=dist_from_df(setdf,False)
        csvname=csvlist[0].split('/')[-1].removesuffix('.csv')
        
        (posdf,box)=load_datafile_from_csv(csvname)

        depthlist=[]
        for index, row in posdf.iterrows():
            if row['type']=='O':
                path=double_recursive_fun(posdf,index)
                if path is not None:
                    depthlist.append(path)

                # path=recursive_fun(posdf,index)
                # if path is not None:
                #     depthlist.append(len(path))
            
        
        print(np.mean(depthlist))
        print(np.min(depthlist))
        print(np.max(depthlist))
        #nt.plot_multi_distribution(df)
        
        #rangeddf =plot_atominrange(setdf,10,'H')