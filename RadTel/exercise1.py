import os
import astropy.constants.codata2014 as C
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 1, figsize=(5, 5))

#https://services.swpc.noaa.gov/text/solar_radio_flux.txt
#10.05.2025 @ 1415
S = 91.7*10**-22
A = np.pi * (4.5/2)**2
ROOT = f'{os.path.dirname(os.path.realpath(__file__))}/'
T_Beam = 2439.39 #MeasuredBeam Switching; Mean
D_Sun = 0.6
D_Beam = (2.52 + 3.02) / 2 #Coss Scan; FWHM
dilutationFactor = (D_Sun / D_Beam)**2
T_Cross = (1990.27 + 3173.59) / 2 #Coss Scan; Height over Base


def openFile(fileName, header=None):
    data = pd.read_csv(ROOT+fileName, delimiter='\\s+', names=header, skiprows=1)
    return data

def calcCalFact(T_Ambient, axis):
    sampleSize = 85
    calDat = openFile('Calibration_onTime.dat', ['Time(s)', 'Temperature(K)'])
    T_Sky = calDat['Temperature(K)'][1:sampleSize]
    Time_Sky = calDat['Time(s)'][1:sampleSize]
    T_Sky_AVG = T_Sky.median()
    T_Sky_STD = T_Sky.std()

    T_Wall = calDat['Temperature(K)'][-sampleSize:-1]
    Time_Wall = calDat['Time(s)'][-sampleSize:-1]
    T_Wall_AVG = T_Wall.median()
    T_Wall_STD = T_Wall.std()

    axis.plot(calDat['Time(s)'], calDat['Temperature(K)'])
    axis.hlines(T_Sky_AVG-T_Sky_STD, Time_Sky.iloc[0], Time_Sky.iloc[-1])
    axis.hlines(T_Sky_AVG+T_Sky_STD, Time_Sky.iloc[0], Time_Sky.iloc[-1])
    axis.fill_between(Time_Sky, T_Sky_AVG-T_Sky_STD, T_Sky_AVG+T_Sky_STD, color='#C4C4C4', alpha=0.2)

    axis.hlines(T_Wall_AVG-T_Wall_STD, Time_Wall.iloc[0], Time_Wall.iloc[-1])
    axis.hlines(T_Wall_AVG+T_Wall_STD, Time_Wall.iloc[0], Time_Wall.iloc[-1])
    axis.fill_between(Time_Wall, T_Wall_AVG-T_Wall_STD, T_Wall_AVG+T_Wall_STD, color='#AD2536', alpha=0.2)
    calibrationFactor = T_Ambient / (T_Wall_AVG - T_Sky_AVG)
    return calibrationFactor, T_Sky_AVG

def DEGC_to_K(deg_C):
    return deg_C + 273.15

def calcDelT(T, T_Sky, calFact):
    return (T - T_Sky) * calFact

def calcApertureEff(delT):
    return (2 * C.k_B * delT)/(S*A)

def calcTEff(delT, eta):
    T = delT/dilutationFactor
    return T/eta

T_HoheWarte_DEG = 15
T_HoheWarte_K = DEGC_to_K(T_HoheWarte_DEG)
calFact, T_Sky = calcCalFact(T_HoheWarte_K, axes)

delT_Cross = calcDelT(T_Cross, T_Sky, calFact)
eta_Cross = calcApertureEff(delT_Cross)
T_Cross_EFF = calcTEff(delT_Cross, eta_Cross) * dilutationFactor

delT_Beam = calcDelT(T_Beam, T_Sky, calFact)
eta_Beam = calcApertureEff(delT_Beam)
T_Beam_EFF = calcTEff(delT_Beam, eta_Beam) * dilutationFactor

print(f"\ndelta T: \n\tCross-Scan:{delT_Cross:.3f} \n\tBeam-Switching:{delT_Beam:.3f} \n\nAperture efficiencies: \n\tCross-Scan:{eta_Cross:.3f} \n\tBeam-Switching:{eta_Beam:.3f} \n\nEffective Temperatures: \n\tCross-Scan:{T_Cross_EFF:.3f} \n\tBeam-Switching:{T_Beam_EFF:.3f} \n")

#Different delT: Beam for eta | Cross for TEff
eta_Mixed = calcApertureEff(delT_Cross)
T_EFF_Mixed = calcTEff(delT_Cross, eta_Beam) * dilutationFactor
print(f"\nWith differnet delT: \n\tAperture:{eta_Mixed:.3f} \n\tTEff:{T_EFF_Mixed:.3f} \n")

plt.show()

