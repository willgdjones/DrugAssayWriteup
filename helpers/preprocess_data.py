import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import os
import cv2
import pickle
from IPython.display import Image
import re
import shutil
import sys
import cv2



with open('textfiles/IXclean_images.txt','r') as f:
    IXimage_IDs = np.unique(['_'.join(x.split('_')[0:2]) for x in f.read().splitlines()]).tolist()
    
with open('textfiles/PositiveControllist.txt','r') as f:
    positive_control_IDs = np.unique(['_'.join([x.split('_')[0],x.split('_')[2]]) for x in f.read().splitlines()]).tolist()
    
with open('textfiles/NegativeControllist.txt','r') as f:
    negative_control_IDs = np.unique(['_'.join([x.split('_')[0],x.split('_')[2]]) for x in f.read().splitlines()]).tolist()


with open('textfiles/AssayLabels.csv','r') as f:
    g = [x.split(',') for x in f.read().splitlines()][1:]
    IXpositiveIDs = ['_'.join([x[1],x[2]+'{:02d}'.format(int(x[3]))]) for x in g]
    


IDlabeldict = dict((ID,0) for ID in IXimage_IDs)
valid_positiveIDs = list(set(IXimage_IDs).intersection(set(IXpositiveIDs)))
IXnegativeIDs = list(set(IXimage_IDs) - set(valid_positiveIDs))
total_valid_postiveIDs = set(valid_positiveIDs).union(set(positive_control_IDs))





