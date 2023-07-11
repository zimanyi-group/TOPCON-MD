#!/usr/bin/env python

import sys
import numpy as np
import ase
from ase import Atoms
from ase.io import read, write

if __name__ == "__main__":
    for file in sys.argv[1:]:
        try:
            a=read(file,format='espresso-in')
        except:
            print(file)
            continue
        output=file[0:-2]+'lammps'
        write(output,a,format='lammps-data')
