# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 16:22:46 2017

@author: ronnie
"""

import numpy as np
import pandas as pd

data_cols = ('Part ID', 'X RBM', 'Y RBM', 'Z RBM', 'X Digital', 'Y Digital', 'Z Digital', 'Orientation')
sum_cols = ('Part ID', 'Lot ID', 'Wafer', 'Site', \
            'X Offset (mg)', 'Y Offset (mg)', 'Z Offset (mg)', \
            'X Sensitivity (LSB/g)'	, 'Y Sensitivity (LSB/g)', 'Z Sensitivity (LSB/g)')

class Sensor(object):
    
    def __init__(self, df, resolution):
        self.data = pd.DataFrame(columns=data_cols)
        self.summary = pd.DataFrame(columns=sum_cols)
        
        self.summary.loc[0, 'Part ID']  = int(df.ChipID.unique()[-1])
        self.summary.loc[0, 'Lot ID']  = str(df.LotID.unique()[-1])
        self.summary.loc[0, 'Wafer']  = int(df.WaferNumber.unique()[-1])
        self.summary.loc[0, 'Site']  = int(df.Site.unique()[-1])
        
        for n, i in enumerate(df.LastKnownOrient.unique()):
            orient_data = df[df.LastKnownOrient == i]
            self.data.loc[n, 'Part ID']  = int(orient_data.ChipID.unique()[-1])
            self.data.loc[n, 'X Digital']  = int(orient_data.X_digital.unique()[-1])
            self.data.loc[n, 'Y Digital']  = int(orient_data.Y_digital.unique()[-1])
            self.data.loc[n, 'Z Digital']  = int(orient_data.Z_digital.unique()[-1])
            self.data.loc[n, 'Orientation']  = str(orient_data.LastKnownOrient.unique()[-1])

        self.calculate_0g_offset(resolution)
        self.calculate_dig_sensitivity()
    
    def calculate_0g_offset(self, resolution):
        
        for i in ['X', 'Y', 'Z']:
            
            average_offset = self.data[(self.data.Orientation != "[+%s]" % i) |
                                       (self.data.Orientation != "[-%s]" % i)]
                
            offset = (average_offset["%s Digital" % i].mean() / 2**resolution) * 1000
                     
#            self.summary.at[0, "%s Offset (mg)" % i] = offset
            self.summary.loc[0, "%s Offset (mg)" % i] = offset
            
    def calculate_dig_sensitivity(self):
        
        for i in ['X', 'Y', 'Z']:
        
            pos_one_g = self.data[self.data.Orientation == '[+%s]' % i]['%s Digital' % i].item()
            neg_one_g = self.data[self.data.Orientation == '[-%s]' % i]['%s Digital' % i].item()
            
            sens = np.abs(pos_one_g - neg_one_g) / 2
            
#            self.summary.at[0, "%s Sensitivity (LSB/g)" % i] = sens
            self.summary.loc[0, "%s Sensitivity (LSB/g)" % i] = sens
    
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, value):
        self._data = value
        
    @property
    def summary(self):
        return self._summary
    
    @summary.setter
    def summary(self, value):
        self._summary = value

class ZeroGOffset(object):
    
    def __init__(self, data):
        
        # remove non 6Tip data
        self.data = data[(data.LastKnownOrient == '[+Z]') |
                         (data.LastKnownOrient == '[-Z]') |
                         (data.LastKnownOrient == '[+Y]') |
                         (data.LastKnownOrient == '[-Y]') |
                         (data.LastKnownOrient == '[+X]') |
                         (data.LastKnownOrient == '[-X]')]
        
        self.summary = pd.DataFrame(columns=sum_cols)
        
        self.resolution = int((np.log(self.data[self.data.LastKnownOrient == '[+Z]']['Z_digital'].mean())/np.log(2)) + 0.5)
        
        self.sensors = []
        for _ in data.ChipID.unique():
            if _ != 0:
                self.sensors.append(Sensor(self.data[self.data.ChipID == _], self.resolution))
            
        for _ in self.sensors:
            self.summary = self.summary.append(_.summary, ignore_index=True)
        
        self.char = pd.DataFrame()
        
        p = ['X Offset (mg)', 'Y Offset (mg)', 'Z Offset (mg)', \
            'X Sensitivity (LSB/g)'	, 'Y Sensitivity (LSB/g)', 'Z Sensitivity (LSB/g)']
        
        for i in p:
            
            self.char.at["Average", i] = self.summary[i].mean()
            self.char.at["St. Deviation", i] = self.summary[i].std()
            self.char.at["Maximum", i] = self.summary[i].max()
            self.char.at["Minimum", i] = self.summary[i].min()