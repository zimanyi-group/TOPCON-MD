from ovito.io import import_file, export_file
import ovito.modifiers as m#import BondAnalysisModifier, CreateBondsModifier,CoordinationAnalysisModifier,TimeSeriesModifier
import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from matplotlib.animation import FuncAnimation 


def coordinationTimeseries(file,coordList,timestepLabels=[],title=''):

    l=len(coordList)
    # try: 
    pipeline = import_file(file)
    
    numframes=pipeline.source.num_frames

  
    pipeline.modifiers.append(m.SelectTypeModifier(property = 'Particle Type', types = {'Si'}))
    pipeline.modifiers.append(m.CoordinationAnalysisModifier(cutoff = 2, number_of_bins = 200,partial=True))
    pipeline.modifiers.append(m.HistogramModifier(bin_count=200, property='Coordination',only_selected=True))
    #pipeline.modifiers.append(m.TimeSeriesModifier(operate_on='HistogramModifier.Coordination'))
    
    
    ts=np.empty([numframes,l]) 
    t=np.arange(numframes)
    for i in t:
        
        data = pipeline.compute(i)
        # # print(data.objects.Count)
        # for o in data.objects:
        #     print(o)
        
        ea = np.array(data.tables['histogram[Coordination]'].xy())
        e=ea[:,0]
        
        for n in np.arange(l):
            ind=np.argmin(np.abs(e-coordList[n]))
            ts[i,n]=ea[ind][1]

    fig = plt.figure()
    for n in np.arange(l):
        val=ts[:,n]
        lstr=coordList[n]
        plt.plot(t,val,label=lstr)

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
        
        #print(str(xratio)+'*'+str(figwidth))        
        plt.axvline(x=xpos)
        plt.text(x=xpos+xbuff,y=yratio*figheight,s=lbl[1])


    plt.title(title)
    plt.legend(loc='upper left',title='Coordination Number')
    plt.xlabel('Timestep')
    plt.ylabel('Count')
    plt.show()
    # df = pd.DataFrame(e, columns=['word', 'frequency'])
    # df.plot(kind='bar', x='word')
    # print(data.tables['histogram[Coordination]'].xy())
    
    
    
    # except:
    #     print(' b ')
    #print(data.tables['time-series'].xy())
    return ts
    
    
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
        
    
        
def bondAnalysis(file):
    try: 
        pipeline = import_file(os.path.join(iroot,f))
        
        numframes=pipeline.source.num_frames
    
        pipeline.modifiers.append(m.CreateBondsModifier(cutoff = 2))
        pipeline.modifiers.append(m.BondAnalysisModifier(partition=m.BondAnalysisModifier.Partition.ByParticleType,bins = 200))
        
        # Export bond angle distribution to an output text file.
        #export_file(pipeline, 'output/bond_angles.txt', 'txt/table', key='bond-angle-distr', end_frame=1)

        # Convert bond length histogram to a NumPy array and print it to the terminal.
        data = pipeline.compute(numframes-1)
    except:
        i=1
        
    # if numframes < 20:
    #     continue
    print(d + ' with '+str(numframes)+' frames.')
    # if np.isclose(peakLen,realLen,atol=errLen) and np.isclose(peakAngle,realAngle,atol=errAng):


    
    
    angleTables=data.tables['bond-angle-distr'].xy()
    angleTypes=data.tables['bond-angle-distr'].y
    angleBins = angleTables[:,0]
    
    for column, name in enumerate(angleTypes.component_names):
        if name != 'Si-O-Si' and name != 'O-Si-O':
            continue
        
        angleCounts=angleTypes[:,column]
        
        peakAngle=angleBins[np.argmax(angleCounts)]
        angTitle= "Angle distribution for bond types:{} with a max angle of {:.1f}".format(name,peakAngle)
        plt.bar(angleBins,angleCounts)
        plt.title(angTitle)
        plt.xlabel('Angle(Degrees)')
        plt.ylabel('Count')
        plt.show()
    
    
    lenTables=data.tables['bond-length-distr'].xy()
    bondTypes=data.tables['bond-length-distr'].y
    bondBins=lenTables[:,0]
    
    for column, name in enumerate(bondTypes.component_names):
        # if name != 'Si-O-Si' and name != 'O-Si-O':
        #     continue
        bondCounts=bondTypes[:,column]

        peakLen=bondBins[np.argmax(bondCounts)]
        lenTitle= "Bond length distribution for bond types:{} with peak at {:.2f}".format(name,peakLen)
        
        
        # max_y_lim = max(counts) + 500
        # print(max_y_lim)
        # min_y_lim = min(counts)
        
        # plt.ylim(min_y_lim, max_y_lim)
        plt.title(lenTitle)
        plt.xlabel('Length(r’$\AA$’')
        plt.bar(bondBins,bondCounts)
        plt.show()