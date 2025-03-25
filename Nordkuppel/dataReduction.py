import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from astropy.io import fits
from astropy.table import Table
from astropy.modeling import models, fitting
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.integrate import simpson
import os
import pandas as pd

from DataManager import DataManager

#Consts
ROOT = os.path.dirname(os.path.realpath(__file__))
M81 = ROOT+'/data/M81/'
NGC2281 = ROOT+'/data/NGC2281/'

#Get Data
'''
Object Structure of return from DataManager.retrieve():
{'Flat_V': []
,'Flat_R': []
,'Flat_B': []
,'Light_V': []
,'Light_R': []
,'Light_B': []
,'Dark': []
,'Bias': []
}
'''
DM = DataManager()
DM.fetchData(M81)
M81 = DM.retrieve()
DM.clear()

DM.fetchData(NGC2281)
NGC2281 = DM.retrieve().copy()
DM.clear()


print(M81)

