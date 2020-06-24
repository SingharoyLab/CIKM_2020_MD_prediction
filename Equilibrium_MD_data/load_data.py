#!/usr/bin/env python

import numpy as np

def load_data(fName):
    Coordinates=[]
    with open(fName,'r') as inputfile:
        nAtoms = int(inputfile.readline())
        i=0
        Coordinates.append([])
        j=0
        for line in inputfile:
            if line.strip()==str(nAtoms):
                i+=1
                Coordinates.append([])
                j=0
            elif 'VMD' in line:
                continue
            else:
                Coordinates[i].append([])
                Coordinates[i][j]=[float(b) for b in line.split()[1:4]]
                j+=1
    
    t = np.array(Coordinates)
    return t, nAtoms

