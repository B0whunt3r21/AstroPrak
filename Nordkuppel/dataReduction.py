import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from astropy.io import fits
from astropy.table import Table
from astropy.modeling import models, fitting
from matplotlib.patches import Circle
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.integrate import simpson
import os
import pandas as pd

from DataManager import DataManager

#Consts
ROOT = os.path.dirname(os.path.realpath(__file__))
#ROOT = os.getcwd()
M81 = ROOT+'/data/M81/'
NGC2281 = ROOT+'/data/NGC2281/'
GAIN = 1.56

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


# Defining the frames, aka here are all the masterframes for each of the pictures.
flat_v = NGC2281["Flat_V"]
flat_r = NGC2281["Flat_R"]
flat_b = NGC2281["Flat_B"]
light_v = NGC2281["Light_V"]
light_r = NGC2281["Light_R"]
light_b = NGC2281["Light_B"]
dark = NGC2281["Dark"]
bias = NGC2281["Bias"]
# exposure times:
exposure_time_dark = 30
exposure_time_flat_b = 4
exposure_time_flat_v = 4
exposure_time_flat_r = 3.5
exposure_time_light_b = 10
exposure_time_light_v = 5
exposure_time_light_r = 5

# first, remove the Bias from the Dark and scale the Dark down to 1 sek.:
master_dark_1_sek = (dark - bias) / exposure_time_dark

# now lets correct the flats for their respective filters and axposure times:
# for that, just multiply the down scaled dark y the respective exposure times,
# as a last step normalise the flat by dividing by the max(flat):
master_flat_b = ((flat_b - bias) - (master_dark_1_sek * exposure_time_flat_b))
master_flat_b = master_flat_b / np.max(master_flat_b)

master_flat_v = ((flat_v - bias) - (master_dark_1_sek * exposure_time_flat_v))
master_flat_v = master_flat_v / np.max(master_flat_v)

master_flat_r = ((flat_r - bias) - (master_dark_1_sek * exposure_time_flat_r))
master_flat_r = master_flat_r / np.max(master_flat_r)

# now pretty much the same for the light framse:
master_light_b = ((light_b - bias) - (master_dark_1_sek * exposure_time_light_b))
corrected_light_b = master_light_b / np.max(master_light_b)

master_light_v = ((light_v - bias) - (master_dark_1_sek * exposure_time_light_v))
corrected_light_v = master_light_v / np.max(master_light_v)

master_light_r = ((light_r - bias) - (master_dark_1_sek * exposure_time_light_r))
corrected_light_r = master_light_r / np.max(master_light_r)


def hms_to_seconds(hour, minute, second):
    return hour * 3600 + minute * 60 + second

def seconds_to_hms(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    sec = seconds % 60
    return hours, minutes, sec

def dms_to_arcseconds(degrees, arcminutes, arcseconds):
    return degrees * 3600 + arcminutes * 60 + arcseconds

def arcseconds_to_dms(arcseconds):
    degrees = arcseconds // 3600
    arcminutes = (arcseconds % 3600) // 60
    sec = arcseconds % 60
    return degrees, arcminutes, sec

reference_star_1_pix = (1466, 1758)
reference_star_2_pix = (392, 1691)
reference_star_3_pix = (950, 1372)
reference_star_4_pix = (1260, 1196)
reference_star_5_pix = (246, 1036)
reference_star_6_pix = (int(np.mean([1626.8, 1641.7])), int(np.mean([942.3, 928.4])))
reference_star_7_pix = (int(np.mean([876.58, 883.14])), int(np.mean([637.87, 631.86])))
reference_star_8_pix = (int(np.mean([1675.69, 1684.49])), int(np.mean([573.26, 562.85])))
reference_star_9_pix = (int(np.mean([717.6, 731.9])), int(np.mean([409.0, 396.0])))

# alle referenzterne sky coordinate werte:
# originalerweise RA [hh:mm:ss]stunden, DEC [dd:mm:ss]degree
# neue version: RA [seconds], DEC [arcseconds]
reference_star_1_coord = (hms_to_seconds(6, 48, 48.52), dms_to_arcseconds(41, 11, 28.4))
reference_star_2_coord = (hms_to_seconds(6, 47, 55.57), dms_to_arcseconds(41, 11, 10.0))
reference_star_3_coord = (hms_to_seconds(6, 48, 22.51), dms_to_arcseconds(41, 8, 3.9))
reference_star_4_coord = (hms_to_seconds(6, 48, 37.54), dms_to_arcseconds(41, 6, 21.1))
reference_star_5_coord = (hms_to_seconds(6, 47, 47.56), dms_to_arcseconds(41, 5, 8.7))
reference_star_6_coord = (hms_to_seconds(6, 48, 55.36), dms_to_arcseconds(41, 3, 49.3))
reference_star_7_coord = (hms_to_seconds(6, 48, 17.94), dms_to_arcseconds(41, 1, 15.9))
reference_star_8_coord = (hms_to_seconds(6, 48, 57.06), dms_to_arcseconds(41, 0, 25.1))
reference_star_9_coord = (hms_to_seconds(6, 48, 10.05), dms_to_arcseconds(40, 59, 10.1))

# alle neuen sterne in Pixelwerten (X,Y): 1bis7 von Elias, 8bis14 von Philipp
new_star_1_pix = (int(np.mean([1751.46, 1759.07])), int(np.mean([875.53, 876.92])))
new_star_2_pix = (int(np.mean([1825.6, 1844.0])), int(np.mean([548.1, 530.9])))
new_star_3_pix = (int(np.mean([1393.1, 1404.0])), int(np.mean([435.7, 424.9])))
new_star_4_pix = (int(np.mean([1407.7, 1417.09])), int(np.mean([864.95, 856.15])))
new_star_5_pix = (int(np.mean([1152.68, 1164.98])), int(np.mean([249.38, 237.18])))
new_star_6_pix = (int(np.mean([449.66, 457.14])), int(np.mean([640.46, 632.23])))
new_star_7_pix = (int(np.mean([329.2, 345.6])), int(np.mean([337.8, 318.8])))
new_star_8_pix = (1073, 1024)
new_star_9_pix = (365, 1476)
new_star_10_pix = (655, 1857)
new_star_11_pix = (1212, 1897)
new_star_12_pix = (1808, 1496)
new_star_13_pix = (465, 1266)
new_star_14_pix = (1755, 1299)

#Circle midpoints (x, y) from reference stars
stars_ref = [
    reference_star_1_pix
    ,reference_star_2_pix
    ,reference_star_3_pix
    ,reference_star_4_pix
    ,reference_star_5_pix
    ,reference_star_6_pix
    ,reference_star_7_pix
    ,reference_star_8_pix
    ,reference_star_9_pix
]
    #--------------------
stars_new = [
    new_star_1_pix
    ,new_star_2_pix
    ,new_star_3_pix
    ,new_star_4_pix
    ,new_star_5_pix
    ,new_star_6_pix
    ,new_star_7_pix
    ,new_star_8_pix
    ,new_star_9_pix
    ,new_star_10_pix
    ,new_star_11_pix
    ,new_star_12_pix
    ,new_star_13_pix
    ,new_star_14_pix
]

#Circle sizes
radii_ref = [
    10
    ,9
    ,8
    ,15
    ,7
    ,11
    ,7
    ,10
    ,10
]
    #-----
radii_new = [
    10
    ,15
    ,12
    ,12
    ,15
    ,10
    ,15
    ,15
    ,15
    ,15
    ,10
    ,10
    ,12
    ,11
]

#Visualization
def plot_marked_stars(image, centers, radii):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap = 'gray', vmin = np.percentile(image, 5), vmax = np.percentile(image, 95))
    
    # Overlay circular patches
    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, color='red', fill=False, linewidth=0.5)
        ax.add_patch(circle)
        ax.text(center[0]+radius, center[1]+radius, f"{i+1}.", color='yellow', fontsize=10, ha='left', va='bottom', 
                bbox=dict(facecolor='black', alpha=0.1, edgecolor='none'))
    
    ax.set_title("Selected Stars")
    plt.show()


def calcMag(fluxes, fluxes_ref, mag_ref):
    mags = []
    for i, f_ref in enumerate(fluxes_ref):
        temp = []
        for f in fluxes:
            mag_new = -2.5*np.log10(np.divide(f, f_ref)) + mag_ref[i]
            temp.append(mag_new)
        mags.append(np.average(temp))
    return mags



def correctFlux(fluxes_ref, expTime):
    return np.divide(np.multiply(fluxes_ref, GAIN), expTime)


mag_ref = pd.DataFrame(columns=["B_mag", "V_mag", "R_mag"],
                       data= np.array([
                           [13.96, 12.71, 12.35]
                           ,[13.63, 12.92, 12.75]
                           ,[15.63, 14.53, 14.12]
                           ,[10.38, 10.14, 10.21]
                           ,[14.26, 13.6, 13.44]
                           ,[11.97, 11.4, 11.3]
                           ,[14.48, 13.76, 13.55]
                           ,[14.25, 13.49, 13.28]
                           ,[11.39, 10.92, 10.85]
                           ]))

plot_marked_stars(corrected_light_v, stars_ref, radii_ref)


fluxes_ref = DM.circularSelection(corrected_light_v, stars_ref, radii_ref)
fluxes_new = DM.circularSelection(corrected_light_v, stars_new, radii_new)

fluxes_ref_Corrected = correctFlux(fluxes_ref, exposure_time_light_v)
fluxes_new_Corrected = correctFlux(fluxes_new, exposure_time_light_v)


mag = calcMag(fluxes_new_Corrected, fluxes_ref_Corrected, mag_ref['V_mag'])
print(mag)


