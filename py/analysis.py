from ovito.io import import_file, export_file
import ovito.modifiers as m#import BondAnalysisModifier, CreateBondsModifier,CoordinationAnalysisModifier,TimeSeriesModifier
import glob, os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import lfilter
from scipy.signal import savgol_filter
import scipy.optimize as so

from matplotlib.animation import FuncAnimation 
from itertools import cycle
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

lines = ["-","-.","--","-",":"]
colors = ["r","g","b","y","c"]

##### This modifier function computes: ####
#
# **(1)** the total coordination number of **TypeA** within the specified cutoff radius
# **(2)** the partial coordination number of **TypeB** around **TypeA** 
# and **(3)** the atom fraction of **TypeB**

from ovito.data import *
import numpy as np
from ovito.modifiers import *

def modify(frame: int, data: DataCollection, typeA = 2, typeB = 1, cutoff_radius = 3.6):
    
    data.apply(SelectTypeModifier(types = {typeA}))
    data.apply(ComputePropertyModifier(output_property = f'Type{typeA}-Type{typeB}-coord', only_selected = True, cutoff_radius = cutoff_radius, neighbor_expressions = (f'ParticleType == {typeB}',)))    
    data.apply(ComputePropertyModifier(output_property = f'Type{typeA}-coord', only_selected = True, cutoff_radius = cutoff_radius, neighbor_expressions = ("1",)))  
    
    print(f"Average Type{typeA} coordination number: {np.sum(data.particles[f'Type{typeA}-coord']/data.attributes['SelectType.num_selected']):.2f}")
    print(f"Average Type{typeA}-Type{typeB} coordination: {np.sum(data.particles[f'Type{typeA}-Type{typeB}-coord']/data.attributes['SelectType.num_selected']):.2f}")
    
    data.apply(SelectTypeModifier(types = {typeB}))
    print(f"Fraction of Type{typeB} atoms: {data.attributes['SelectType.num_selected.2']/data.particles.count:.2f}")
    
def coordinationTimeseries(folderList,coordList):
    pipelineList=[]
    numCoordNumbers=len(coordList)

    # os.chdir(folder)
    for folder in folderList:

        datafile=''
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith(".dump") and file != "bonds.dump":#this uses the folder's dump file
                    datafile=os.path.join(root,file)
            
        try:
                    
            pipeline = import_file(datafile)
            
            #numframes=pipeline.source.num_frames

            # pipeline.modifiers.append(m.CreateBondsModifier(cutoff = 2))
            # pipeline.modifiers.append(m.BondAnalysisModifier(partition=m.BondAnalysisModifier.Partition.ByParticleType,bins = 200))
            
            # Export bond angle distribution to an output text file.
            #export_file(pipeline, 'output/bond_angles.txt', 'txt/table', key='bond-angle-distr', end_frame=1)

            # Convert bond length histogram to a NumPy array and print it to the terminal.
            #data = pipeline.compute()
            pipelineList.append(pipeline)
            
        except Exception as e:
            print(folder)
            print(datafile) 
            print(e)
        
    

    numSamples = len(pipelineList)
    tsData=np.empty(numSamples,dtype=object)
    for p in range(numSamples):
        pipeline=pipelineList[p]
        # if len(pipelineList) > 1:
        #     curColor = next(colorcycler)
            
        #curLine = next(linecycler)
        numframes=pipeline.source.num_frames

    
        pipeline.modifiers.append(m.SelectTypeModifier(property = 'Particle Type', types = {'Si'}))
        pipeline.modifiers.append(m.ComputePropertyModifier(output_property = f'Coordination', only_selected = True, cutoff_radius = 2, neighbor_expressions = (f'ParticleType == 2',)))
        pipeline.modifiers.append(m.HistogramModifier(bin_count=200, property='Coordination',only_selected=True))

        
        vals=np.empty([numCoordNumbers,numframes]) 
        t=np.arange(numframes)
        
        for i in t:
            
            data = pipeline.compute(i)
            # # print(data.objects.Count)
            # for o in data.objects:
            #     print(o)
            
            hist = np.array(data.tables['histogram[Coordination]'].xy())
            rawCoord=hist[:,0]

            for n in np.arange(numCoordNumbers):
                ind=np.argmin(np.abs(rawCoord-coordList[n]))
                vals[n,i]=hist[ind][1]
        
        tsData[p]=(t,vals)
        
    return tsData

def plotTimeSeries(data,coordList,reduction=0,timestepLabels=[],title=''):
    fig = plt.figure()
    linecycler = cycle(lines)
    colorcycler = cycle(colors)
    numCoordNumbers=len(coordList)
    numSamples=data.size

    for s in np.arange(numSamples):
        if numSamples > 1:
            curColor = next(colorcycler)
        else:
            curColor = colors[s%4]
            
        for n in np.arange(numCoordNumbers):
            num=coordList[n]
            d=data[s]
            t=d[0]
            val=d[1][n]
            #if there's only one pipeline then we change colors but if more then we change style
            if numCoordNumbers > 1:
                curLine=lines[n%4]
            else:
                curLine='-'
                
            
            
            
            # print(t)
            # print(val)
            lstr=coordList[n]
            # h=15
            # b=[1.0/h]*h
            # a=1
            # yy=lfilter(b,a,val)
            w=savgol_filter(val,25,3)
            # vv=so.curve_fit(sillyBilly,t,val)
            # print(vv[0][1])
            # print(vv[0][2])
            # #plt.plot(t,sillyBilly(t,val,vv[0][1],vv[0][2]),color=curColor)

            plt.plot(t,w,curLine,color=curColor,label=lstr)

    ###### Plot the different labels for time regions
    xaxislen=fig.gca().get_xlim()[1]
    figwidth=fig.get_figwidth()
    figwidth,figheight=fig.canvas.get_width_height()
    yratio=.15
    xbuff=15#pixels of buffer from text to vertical line
    
    #Ratio of all text's height on figure

    for lbl in timestepLabels:
        xpos=lbl[0]
        xratio=xpos/xaxislen
                
        plt.axvline(x=xpos)
        plt.text(x=xpos+xbuff,y=yratio*figheight,s=lbl[1])


    plt.title(title)
    #plt.ylim(bottom=0,top=500)
    legend_elements =[
                        Line2D([0],[0],color=colors[0],linestyle='-',label='0 added H'),
                        Line2D([0],[0],color=colors[1],linestyle='-',label='25 added H'),
                        Line2D([0],[0],color=colors[2],linestyle='-',label='50 added H'),
                        Line2D([0],[0],color=colors[3],linestyle='-',label='100 added H')]
                        # mpatches.Patch(color='none',label='Coordination Number'),
                        # Line2D([0],[0],color=colors[0],linestyle=lines[0],label='4'),
                        # Line2D([0],[0],color=colors[0],linestyle=lines[3],label='3'),
                        # Line2D([0],[0],color=colors[0],linestyle=lines[2],label='2')]
                    #   Line2D([0],[0],color=colors[0],linestyle=lines[1],label='1'),
                    #   Line2D([0],[0],color=colors[0],linestyle=lines[0],label='0')]
    plt.legend(bbox_to_anchor=(1.05,1),handles=legend_elements,loc='upper left',handlelength=3,title='Sample')
    plt.xlabel('Time Step')
    plt.ylabel('Count')

    
    plt.show()
    
def rdfTimeseries(file,range,out):
    # fig = plt.figure() 
    
    # # marking the x-axis and y-axis
    # axis = plt.axes(xlim =(0, 4)) 
    
    # # initializing a line variable
    # line, = axis.plot([], [], lw = 3) 


    pipeline = import_file(file)
    
    numframes=pipeline.source.num_frames


    # Print the list of input particle types.
    # They are represented by ParticleType objects attached to the 'Particle Type' particle property.
    # for t in pipeline.compute().particles.particle_types.types:
    #     print("Type %i: %s" % (t.id, t.name))

    # Calculate partial RDFs:
    pipeline.modifiers.append(m.CoordinationAnalysisModifier(cutoff=5.0, number_of_bins=100, partial=True))
    for i in np.arange(numframes):
            
        # Access the output DataTable:
        rdf_table = pipeline.compute(i).tables['coordination-rdf']

        # The y-property of the data points of the DataTable is now a vectorial property.
        # Each vector component represents one partial RDF.
        rdf_names = rdf_table.y.component_names
        t=rdf_table.xy()[:,0]
        # Print a list of partial g(r) functions.
        for component, name in enumerate(rdf_names):
            # print("g(r) for pair-wise type combination %s:" % name)
            plt.plot(t,rdf_table.y[:,component],label=name)
        numstr=''
        if i < 10:
            numstr+='00'+str(i) 
        elif i < 100:
            numstr='0'+str(i) 
        else:
           numstr=str(i)  
        plt.legend(loc='upper left',title='Pairs')
        plt.xlabel('Pair separation distance(angstroms)')
        plt.ylabel('g(r)')
        plt.ylim(bottom=0,top=10)
        plt.xlim(left=2,right=3)
        plt.title('RDF - timestep ' + numstr)
        plt.savefig(out+'coordts-'+numstr)
        plt.clf()


    # The DataTable.xy() method yields everthing as one combined NumPy table.
    # This includes the 'r' values in the first array column, followed by the
    # tabulated g(r) partial functions in the remaining columns. 
    # print(rdf_table.xy()[:,0])

    
    # def init(file):
    #     return file
    
    # def animate(i):
    #     return i
              
def bondAnalysis(folder,plot):
    try: 
        datafile=''
        # os.chdir(folder)
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith(".data"):#uses a data file not log file
                    datafile=os.path.join(root,file)
    
    
        pipeline = import_file(datafile)
        
        #numframes=pipeline.source.num_frames

        pipeline.modifiers.append(m.CreateBondsModifier(cutoff = 2.7))
        pipeline.modifiers.append(m.BondAnalysisModifier(partition=m.BondAnalysisModifier.Partition.ByParticleType,bins = 200))
        
        # Export bond angle distribution to an output text file.
        #export_file(pipeline, 'output/bond_angles.txt', 'txt/table', key='bond-angle-distr', end_frame=1)

        # Convert bond length histogram to a NumPy array and print it to the terminal.
        data = pipeline.compute()
    except Exception as e: 
        print(datafile)
        print(e)
        

    # if np.isclose(peakLen,realLen,atol=errLen) and np.isclose(peakAngle,realAngle,atol=errAng):


    
    
    angleTables=data.tables['bond-angle-distr'].xy()
    angleTypes=data.tables['bond-angle-distr'].y
    angleBins = angleTables[:,0]
    
    outlist=[]
    
    for column, name in enumerate(angleTypes.component_names):
        angleCounts=angleTypes[:,column]
        if np.sum(angleCounts) == 0:
            continue
        
        
        peakAngle=angleBins[np.argmax(angleCounts)]
        angTitle= "Angle distribution for bond types: {} with a max angle of {:.1f}".format(name,peakAngle)
        outlist.append(angTitle)
        if plot:
            plt.bar(angleBins,angleCounts)
            plt.title(angTitle)
            plt.xlabel('Angle(Degrees)')
            plt.ylabel('Count')
            plt.show()
    
    
    lenTables=data.tables['bond-length-distr'].xy()
    bondTypes=data.tables['bond-length-distr'].y
    bondBins=lenTables[:,0]
    
    for column, name in enumerate(bondTypes.component_names):

        bondCounts=bondTypes[:,column]
        if np.sum(bondCounts)==0:
            continue
        peakLen=bondBins[np.argmax(bondCounts)]
        lenTitle= "Bond length distribution for bond types: {} with peak at {:.2f}".format(name,peakLen)
        outlist.append(lenTitle)
        if plot:
            # max_y_lim = max(counts) + 500
            # print(max_y_lim)
            # min_y_lim = min(counts)
            
            # plt.ylim(min_y_lim, max_y_lim)
            plt.title(lenTitle)
            plt.xlabel('Length(r’$\AA$’')
            plt.bar(bondBins,bondCounts)
            plt.show()
        
    for s in outlist:
        print(s)