import os
import pandas as pd
import numpy as np
from astropy.io import fits


class DataManager():
    def __init__(self):
        self.Flats_V = pd.DataFrame(columns=['Nr', 'Flat_V'])
        self.Flats_R = pd.DataFrame(columns=['Nr', 'Flat_R'])
        self.Flats_B = pd.DataFrame(columns=['Nr', 'Flat_B'])
        self.Light_V = pd.DataFrame(columns=['Nr', 'Light_V'])
        self.Light_R = pd.DataFrame(columns=['Nr', 'Light_R'])
        self.Light_B = pd.DataFrame(columns=['Nr', 'Light_B'])
        self.Darks = pd.DataFrame(columns=['Nr', 'Dark'])
        self.Biases = pd.DataFrame(columns=['Nr', 'Bias'])


    #Clear
    def clear(self):
        self.Flats_B.drop(labels=self.Flats_B.index, axis=0, inplace=True)
        self.Flats_R.drop(labels=self.Flats_R.index, axis=0, inplace=True)
        self.Flats_V.drop(labels=self.Flats_V.index, axis=0, inplace=True)
        self.Light_B.drop(labels=self.Light_B.index, axis=0, inplace=True)
        self.Light_R.drop(labels=self.Light_R.index, axis=0, inplace=True)
        self.Light_V.drop(labels=self.Light_V.index, axis=0, inplace=True)
        self.Darks.drop(labels=self.Darks.index, axis=0, inplace=True)
        self.Biases.drop(labels=self.Biases.index, axis=0, inplace=True)
        self.Path = ""

    
    #Stacks the Images inside the Frames
    def stack(self, frame):
        stacked_Frame = np.median(np.stack(frame.iloc[:, 1].to_numpy()), axis=0)
        return stacked_Frame

    #Forward data
    def retrieve(self):
        return {'Flat_V': self.stack(self.Flats_V.copy())
                ,'Flat_R': self.stack(self.Flats_R.copy())
                ,'Flat_B': self.stack(self.Flats_B.copy())
                ,'Light_V': self.stack(self.Light_V.copy())
                ,'Light_R': self.stack(self.Light_R.copy())
                ,'Light_B': self.stack(self.Light_B.copy())
                ,'Dark': self.stack(self.Darks.copy())
                ,'Bias': self.stack(self.Biases.copy())
        }
    

    def openFit(self, path, fileName):
        with fits.open(path+fileName) as fit:
            data = fit[0].data
        return data

    
    #Sorts files' data into corresponding DataFrames
    def fetchData(self, path):
        files = [file for file in os.listdir(path) if file.endswith('fits')]

        #Split Filenames for Sorting and def Sorting-Frame
        tempArr = []

        for file in files:
            nr = file.split('_')[0]
            if file.endswith('fits') == True:
                tempArr.append(int(nr))
            else:
                tempArr.append(None)

        #Lookup DF Nr-File
        fileFrame = pd.DataFrame({'Nr': tempArr, 'Files': files})
        fileFrame.sort_values(['Nr', 'Files'], inplace=True)

        #Sort Dark, Bias, Light and Flat Files
        for idx, row in fileFrame.iterrows():
            nr = row['Nr']
            fileName = row['Files']
            fitData = self.openFit(path, fileName)

            if fileName.lower().find('bias') != -1:
                self.Biases.loc[len(self.Biases)] = [nr, fitData]

            elif fileName.lower().find('dark') != -1:
                self.Darks.loc[len(self.Darks)] = [nr, fitData]

            elif fileName.lower().find('flat') != -1:
                if fileName.lower().find('v') != -1:
                    self.Flats_V.loc[len(self.Flats_V)] = [nr, fitData]
                elif fileName.lower().find('r') != -1:
                    self.Flats_R.loc[len(self.Flats_R)] = [nr, fitData]
                elif fileName.lower().find('b') != -1:
                    self.Flats_B.loc[len(self.Flats_B)] = [nr, fitData]
                else:
                    print("Missmatch")

            elif fileName.lower().find('light') != -1:
                if fileName.lower().find('v') != -1:
                    self.Light_V.loc[len(self.Light_V)] = [nr, fitData]
                elif fileName.lower().find('r') != -1:
                    self.Light_R.loc[len(self.Light_R)] = [nr, fitData]
                elif fileName.lower().find('b') != -1:
                    self.Light_B.loc[len(self.Light_B)] = [nr, fitData]
                else:
                    print("Missmatch")
                
            else:
                print("Missmatch")

        files.clear()
        tempArr.clear()


