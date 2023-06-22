import ReadLAMMPS_2 as rl
import sys
import os
import re
import linecache as lc
import atom
import numpy as np
import multiprocessing as mp
import pdb
import time
#import Atom_Manip as AM
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
matplotlib.use('Agg')



def distform(coords1,coords2,dims):
	delta = np.zeros([3])
	for i in range(0,3):
		dx = np.abs(coords1[i]-coords2[i])
		if dx > dims[i]/2:
			delta[i] = dims[i] - dx
		else:
			delta[i]=dx
	return np.sqrt((delta[0])**2 + (delta[1])**2 + (delta[2])**2)

def generate_pairs(OHdists,hindval,timesteps):
	singleHdata=OHdists[:,hindval,:]
	numsteps=len(timesteps)
	print(singleHdata)
	pair_list=[]
	temparray=[]
 
 
	for i in range(0,numsteps):
		current_O=singleHdata[i,1]
		current_dist=singleHdata[i,2]
		current_time=timesteps[i]
		print(current_O)
		if i==0:
			prev_O=singleHdata[0,1]

			continue

		if current_O == prev_O:
			temparray.append([current_O,current_dist,current_time])

		else:
			pair_list.append(temparray)
			temparray=[]


		prev_O=current_O
	return pair_list

		
			




dirname="/home/agoga/documents/code/topcon-md/data/SiOxNEB-NOH.dump"

dir = os.fsencode(dirname)

sim = rl.Read_Dump(dir)
numsteps = 0
flag = True
dt = 1.5 * 10**-6 #ns
dumpdt = 10000 #num of timesteps between dump writes

timesteplist=[]
while flag==True:
    
	try:

		sim.Update('next')
		numsteps+=1
		timesteplist.append(sim.timestep)
	except:
		flag=False

print('here')
f1=rl.Read_Dump(dir)
dims=[f1.box[0][1]-f1.box[0][0], f1.box[1][1]-f1.box[1][0], f1.box[2][1]-f1.box[2][0]]
flag=0







OHcutoff= 1.05*2.437
OHdists=[]
targetTS=1


numsteps=5
for i in range(0,numsteps):
	for at in f1.atoms:
		if at.type == 2:
			x1,y1,z1=at.coords

			for at2 in f1.atoms:
				if at2.type == 2:
					x2,y2,z2=at2.coords
					tempdist=distform(at.coords,at2.coords,dims)
					if tempdist<OHcutoff:
						OHdists.append([at.id, at2.id, tempdist, x1, y1, z1, f1.timestep])


	print('{} out of {} timesteps'.format(str(i),len(timesteplist)))
	print(f1.timestep)
	if f1.timestep == targetTS:
		break
	f1.Update('next')
 
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
OHdists=np.array(OHdists)
with open('andrew-1900-4-6.txt', 'w') as f:
	f.write('O-id, H-id, OH-dist, Ox, Oy, Oz, timestep\n')
	for i in range(0,len(OHdists)):
		#f.write('{}, {}, {} \n'.format(totalH[i],mobileH[i],OH[i])
			f.write('{},{},{},{},{},{},{}\n'.format(OHdists[i,0],OHdists[i,1],OHdists[i,2],OHdists[i,3],OHdists[i,4],OHdists[i,5],OHdists[i,6]))



#analyze OHdists to gather OH complex lifetimes
	#check if H stays next to the same O DONE
	#record how long OH complex is together 
	#check the mean separation distance of that complex
	#check the stdev of that separation distance
	#plot each OH pair distance vs time to get

#write function to generate all pair lists for a single H DONE
#loop through H's
#plot
'''
pairs_out=generate_pairs(OHdists,0,timesteplist)
print(pairs_out)
'''