import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import get_body, EarthLocation


ROOT = f'{os.path.dirname(os.path.realpath(__file__))}/'

figDrift, driftAxes = plt.subplots(1, 2, figsize=(10, 5))
figPolar, polarAxes = plt.subplots(1, 1, figsize=(5, 5))
polarAx = figPolar.add_subplot(111, projection='polar')
figVel, velAx = plt.subplots(1, 1, figsize=(5, 5))


def openFile(fileName, header=None, dellim='\\s+'):
    data = pd.read_csv(ROOT+fileName, delimiter=dellim, names=header, skiprows=1, na_values=["NaN", "nan", "None", ""])
    return data



def spiralArms(axis):
    data = openFile('spiralArms.csv', ['GLON', 'VLSR_1', 'VLSR_2', 'VLSR_3', 'VLSR_4', 'VLSR_5', 'VLSR_6'], ',')

    for n in range(1, 7):
        R_n = data[f'VLSR_{n}'].fillna(0) / (14.82 * np.sin(2 * np.radians(data['GLON'])))
        polar = np.column_stack((data[f'VLSR_{n}'].fillna(0) / 180 * np.pi, R_n)) 
        #axis.plot(data['GLON'], polar[:, 1], linestyle='', marker='.', c='y')
        axis.plot(polar[:, 0], polar[:, 1], linestyle='', marker='.', c='y')

    #axis.set_xlim([5, 240])
    axis.set_xlabel('GLON (DEG)')
    axis.set_ylim([0, 11])
    axis.set_ylabel('Radius (kPc)')
    axis.set_theta_zero_location('N')



def velocity(axis):
    data = openFile('rotCurve.csv', ['GLON', 'VLSR', 'raw'], ',')
    meanVel = data['VLSR'].iloc[1:-15].mean()
    axis.set_xlabel('GLON')
    axis.set_ylabel('VLSR')
    axis.plot(data['GLON'], data['raw'], linestyle=':', label='raw', color='gray')
    axis.plot(data['GLON'], data['VLSR'], label='shifted', color='blue')
    axis.hlines(meanVel, data['GLON'].iloc[0], data['GLON'].iloc[-1], color='red', linestyle=':')
    axis.legend()



def readDriftScan(fileName):
    chunks = []

    with open(ROOT+fileName, 'r') as f:
        next(f)
        for line in f:
            values = line.strip().split()
            metadata = values[:18]
            bins = int(metadata[-1])

            data = np.array(values[18:18+bins], dtype=float)

            chunks.append({
                "LAT": float(metadata[0]),
                "LON": float(metadata[1]),
                "YEAR": int(metadata[2]),
                "MONTH": int(metadata[3]),
                "DAY": int(metadata[4]),
                "LST": float(metadata[5]),
                "AZ": float(metadata[6]),
                "EL": float(metadata[7]),
                "RA": float(metadata[8]),
                "DE": float(metadata[9]),
                "GLAT": float(metadata[10]),
                "GLON": float(metadata[11]),
                "FIRST_FREQ(MHz)": float(metadata[12]),
                "LAST_FREQ(MHz)": float(metadata[13]),
                "REF_FREQ(MHz)": float(metadata[14]),
                "VRAD0(km/s)": float(metadata[15]),
                "VRAD1(km/s)": float(metadata[16]),
                "BINS": bins,
                "Amplitudes(K)": data
            })

    return pd.DataFrame(chunks)


def format_LST(lst):
    hours = int(lst)
    minutes = int((lst - hours) * 60)
    seconds = int(((lst - hours) * 60 - minutes) * 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def temporalShift(data):
    allObsTimes = []
    shiftedRa = []

    for n in range(len(data)):
        times = [f"{int(data[n]['YEAR'].iloc[row])}-{int(data[n]['MONTH'].iloc[row]):02d}-{int(data[n]['DAY'].iloc[row]):02d}T{format_LST(data[n]['LST'].iloc[row])}" for row in range(len(data[n]['LST']))]
        allObsTimes.append(Time(times, format="isot"))
    refTime = min([obs[-1] for obs in allObsTimes])

    for n in range(len(data)):
        timeStrings = [f"{int(data[n]['YEAR'].iloc[row])}-{int(data[n]['MONTH'].iloc[row]):02d}-{int(data[n]['DAY'].iloc[row]):02d}T{format_LST(data[n]['LST'].iloc[row])}" for row in range(len(data[n]['LST']))]
        obsTimes = Time(timeStrings, format="isot")

        obsLat, obsLon = data[n]["LAT"].iloc[0], data[n]["LON"].iloc[0]
        location = EarthLocation(lat=obsLat, lon=obsLon, height=420)
        moonPositions = get_body('moon', obsTimes, location).transform_to('icrs')
        reference_moon = get_body('moon', refTime, location).transform_to('icrs')

        raShift = np.subtract(reference_moon.ra, moonPositions.ra)
        raShift_deg = raShift.to(u.deg).value
        shiftedRa.append(data[n]['RA'].values + raShift_deg)

    return shiftedRa


def interpolateToLongest(data, shiftedRa):
    maxLen = max(len(shiftedRa[n]) for n in range(len(data)))
    interpolatedData = []
    newAxis = []

    for n in range(len(data)):
        spect = np.mean(data[n]['Amplitudes(K)'].T, axis=0)

        ogAxis = np.linspace(shiftedRa[n][0], shiftedRa[n][-1], len(spect))
        newAxis.append(np.linspace(ogAxis[0], ogAxis[-1], maxLen))

        interp_func = interp1d(ogAxis, spect, kind='linear', fill_value="extrapolate")
        interpolatedData.append(interp_func(newAxis[n]))

    return newAxis, interpolatedData


def driftScan(axis):
    filenames = ['DriftScan_Moon.dat', 'DriftScan_Moon_2.dat']
    AllData = [readDriftScan(i) for i in filenames]

    shiftedRa = temporalShift(AllData)
    interpRa, interpCont = interpolateToLongest(AllData, shiftedRa)

    for n in range(len(AllData)):
        mean_data = np.mean(AllData[n]['Amplitudes(K)'], axis=0)
        max_data = np.max(np.vstack(AllData[n]['Amplitudes(K)'].to_numpy()), axis=0)
        
        spectAxis = np.linspace(AllData[n]['FIRST_FREQ(MHz)'].iloc[0], AllData[n]['LAST_FREQ(MHz)'].iloc[-1], len(mean_data))
        axis[0].fill_between(spectAxis, mean_data, np.min(mean_data), alpha=1, label='Mean')
        axis[0].plot(spectAxis, max_data, color='red', linewidth=0.5, label='Max')
        axis[0].set_title(f"Raw Spectrum")
        axis[0].set_xlabel("Frequency (MHz)")
        axis[0].set_ylabel("Intensity (Kelvin)")
        axis[0].legend()

        continuumAxis = np.linspace(interpRa[n][0], interpRa[n][-1], len(interpCont[n]))
        axis[1].plot(continuumAxis, interpCont[n], linestyle='-', marker='.')
        axis[1].set_title(f"Continuum")
        axis[1].set_xlabel("Right ascension (Hours)")
        axis[1].set_ylabel("Intensity (Kelvin)")



driftScan(driftAxes)
spiralArms(polarAx)
velocity(velAx)

plt.tight_layout()
plt.show()


figPolar.savefig(ROOT + 'img/SpiralArms.svg')
figVel.savefig(ROOT + 'img/RotationCurve.svg')
figDrift.savefig(ROOT + 'img/LunarDriftScan.svg')


