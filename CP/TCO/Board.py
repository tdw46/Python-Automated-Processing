# -*- coding: utf-8 -*-

import itertools
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import pymysql as pms
import os
import getpass
import os.path

data_cols = ('Part ID', 'Reading', 'Temperature (C)', \
             'X RBM', 'Y RBM', 'Z RBM', \
             'X Digital', 'Y Digital', 'Z Digital', 'Orientation')
temp_cols = ('Part ID', 'Temperature', \
             'X Sensitivity (fF/g)', 'Y Sensitivity (fF/g)', 'Z Sensitivity (fF/g)', \
             'X Sensitivity (LSB/g)', 'Y Sensitivity (LSB/g)', 'Z Sensitivity (LSB/g)', \
             'X Offset (mg)', 'Y Offset (mg)', 'Z Offset (mg)', \
             'X Sensitivity (%/C)', 'Y Sensitivity (%/C)',  'Z Sensitivity (%/C)')
sum_cols = ('Part ID', 'Lot ID', 'Wafer', 'Site', \
            'X TCO (mg/C)', 'Y TCO (mg/C)', 'Z TCO (mg/C)', \
            'X TCS (%/C)', 'Y TCS (%/C)', 'Z TCS (%/C)')

class Sensor(object):
    
    def __init__(self, data, resolution):
        data.reset_index(inplace=True)
        
        self.big_data = data
        
        self.data = pd.DataFrame(columns=data_cols)
        self.temp = pd.DataFrame(columns=temp_cols)
        self.summary = pd.DataFrame(columns=sum_cols)
        
        self.resolution = resolution
        
        self.summary.loc[0, 'Part ID']  = int(data.ChipID.unique()[-1])
        self.summary.loc[0, 'Lot ID']  = str(data.LotID.unique()[-1])
        self.summary.loc[0, 'Wafer']  = int(data.WaferNumber.unique()[-1])
        self.summary.loc[0, 'Site']  = int(data.Site.unique()[-1])
        
        reading_count = -1
        
        for i in range(len(data)):
            
            if i % 6 == 0:
                reading_count = reading_count + 1
            
            self.data.loc[i, 'Part ID']  = int(data.ChipID.unique()[-1])
            self.data.loc[i, 'Reading'] = reading_count
            self.data.loc[i, 'Temperature (C)']  = int(data.BoardTemp[i])
            self.data.loc[i, 'X RBM']  = int(data.X_RBM[i])
            self.data.loc[i, 'Y RBM']  = int(data.Y_RBM[i])
            self.data.loc[i, 'Z RBM']  = int(data.Z_RBM[i])
            self.data.loc[i, 'X Digital']  = int(data.X_digital[i])
            self.data.loc[i, 'Y Digital']  = int(data.Y_digital[i])
            self.data.loc[i, 'Z Digital']  = int(data.Z_digital[i])
            self.data.loc[i, 'Orientation']  = str(data.LastKnownOrient[i])
            
        print(self.data)
            
#        self.calculate_tcs()
        self.calculate_tco()
            
            
    def calculate_tco(self):
        
        for i in self.data.Reading.unique():
            
            self.calculate_1g_offset(self.data[self.data.Reading == i])
            
        for i in ['X', 'Y', 'Z']:
            
            max_temp_off = self.temp[self.temp.Temperature == self.temp.Temperature.max()]["%s Offset (mg)" % i].item()
            min_temp_off = self.temp[self.temp.Temperature == self.temp.Temperature.min()]["%s Offset (mg)" % i].item()
            temp_spread = self.temp.Temperature.max() - self.temp.Temperature.min()
            
            tco = (max_temp_off - min_temp_off) / temp_spread
                
            self.summary.set_value(0, '%s TCO (mg/C)' % i, tco)
    
    def calculate_0g_offset(self, off_data):
        
        off_data.reset_index(inplace=True)
        
        self.temp.set_value(off_data.Reading[0], "Part ID", self.summary["Part ID"].item())
        
        temperature = off_data['Temperature (C)'].unique()[-1]
        self.temp.set_value(off_data.Reading[0], "Temperature", temperature)
        
        for i in ['X', 'Y', 'Z']:
            
            average_offset = off_data[(off_data.Orientation != "[+%s]" % i) |
                                      (off_data.Orientation != "[-%s]" % i)]
                
            offset = (average_offset["%s Digital" % i].mean() / 2**self.resolution) * 1000
                     
            self.temp.set_value(off_data.Reading[0], "%s Offset (mg)" % i, offset)
    
    def calculate_1g_offset(self, off_data):
        
        off_data = off_data[off_data.Orientation == '[+Z]']
        off_data.reset_index(inplace=True)
        
        self.temp.set_value(off_data.Reading[0], "Part ID", self.summary["Part ID"].item())
        
        temperature = off_data['Temperature (C)'].unique()[-1]
        self.temp.set_value(off_data.Reading[0], "Temperature", temperature)
        
        for i in ['X', 'Y', 'Z']:
            
            if i == 'Z':
                expected_offset = 2 ** self.resolution
            else:
                expected_offset = 0
                
            offset = ((off_data["%s Digital" % i].item() - expected_offset) / 2**self.resolution) * 1000
            
            self.temp.set_value(off_data.Reading[0], "%s Offset (mg)" % i, offset)
    
    def calculate_tcs(self):
        
        for i in self.data.Reading.unique():
#            self.calculate_mems_sensitivity(self.data[self.data.Reading == i])
            self.calculate_dig_sensitivity(self.data[self.data.Reading == i])
            
        for i in ['X', 'Y', 'Z']:
            
            # mems_sens
#            max_temp_sens = self.temp[self.temp.Temperature == self.temp.Temperature.max()]["%s Sensitivity (fF/g)" % i].item()
#            min_temp_sens = self.temp[self.temp.Temperature == self.temp.Temperature.min()]["%s Sensitivity (fF/g)" % i].item()
            
            # dig_sens
#            max_temp_sens = self.temp[self.temp.Temperature == self.temp.Temperature.max()]["%s Sensitivity (LSB/g)" % i].item()
#            min_temp_sens = self.temp[self.temp.Temperature == self.temp.Temperature.min()]["%s Sensitivity (LSB/g)" % i].item()
            
#            temp_spread = self.temp.Temperature.max() - self.temp.Temperature.min()
            
            
#            tcs = (max_temp_sens - min_temp_sens) / temp_spread

            tcs = self.temp["%s Sensitivity (%%/C)" % i].max()
                
#            self.summary.set_value(0, '%s TCS (fF/g/C)' % i, tcs)
            
            self.summary.set_value(0, '%s TCS (%%/C)' % i, tcs)
    
    def calculate_mems_sensitivity(self, sens_data):
        
        sens_data.reset_index(inplace=True)

        self.temp.set_value(sens_data.Reading[0], "Part ID", self.summary["Part ID"].item())
        
        temperature = sens_data['Temperature (C)'].unique()[-1]
        self.temp.set_value(sens_data.Reading[0], "Temperature", temperature)
        
        for i in ['X', 'Y', 'Z']:
        
            pos_one_g = sens_data[sens_data.Orientation == '[+%s]' % i]['%s RBM' % i].item()
            neg_one_g = sens_data[sens_data.Orientation == '[-%s]' % i]['%s RBM' % i].item()
            
            sens = np.abs(pos_one_g - neg_one_g) / 2 / (((2 ** 15) / 48) / 1.536)
            
            self.temp.set_value(sens_data.Reading[0], "%s Sensitivity (fF/g)" % i, sens)
            
            if temperature != 25 and sens_data.Reading[0] != 0:
                sens_change = (np.abs(sens - self.temp["%s Sensitivity (fF/g)" % i][0]) / np.abs(temperature - self.temp["Temperature"][0])) * 100
            else:
                sens_change = 0
            
            self.temp.set_value(sens_data.Reading[0], "%s Sensitivity (%%/C)" % i, sens_change)
            
    def calculate_dig_sensitivity(self, sens_data):
    
        sens_data.reset_index(inplace=True)

        self.temp.set_value(sens_data.Reading[0], "Part ID", self.summary["Part ID"].item())
        
        temperature = sens_data['Temperature (C)'].unique()[-1]
        self.temp.set_value(sens_data.Reading[0], "Temperature", temperature)
        
        for i in ['X', 'Y', 'Z']:
        
            pos_one_g = sens_data[sens_data.Orientation == '[+%s]' % i]['%s Digital' % i].item()
            neg_one_g = sens_data[sens_data.Orientation == '[-%s]' % i]['%s Digital' % i].item()
            
            sens = np.abs(pos_one_g - neg_one_g) / 2
            
            self.temp.set_value(sens_data.Reading[0], "%s Sensitivity (LSB/g)" % i, sens)
            
            if temperature != 25 and sens_data.Reading[0] != 0:
                sens_change = ((np.abs(sens - self.temp["%s Sensitivity (LSB/g)" % i][0]) / self.temp["%s Sensitivity (LSB/g)" % i][0] ) / np.abs(temperature - self.temp["Temperature"][0])) * 100
            else:
                sens_change = 0
            
            self.temp.set_value(sens_data.Reading[0], "%s Sensitivity (%%/C)" % i, sens_change)
              
def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    
    return ("%s" % int(100 * y))
            
class TCO(object):
    
    def __init__(self, data):
        
        self.data = data
        self.temp = pd.DataFrame(columns=temp_cols)
        self.summary = pd.DataFrame(columns=sum_cols)
        
        self.resolution = int((np.log(self.data[self.data.LastKnownOrient == '[+Z]']['Z_digital'].mean())/np.log(2)) + 0.5)
        
        self.sensors = []
        for _ in data.ChipID.unique():
            if _ != 0:
                self.sensors.append(Sensor(self.data[self.data.ChipID == _], self.resolution))

        for _ in self.sensors:
            self.summary = self.summary.append(_.summary, ignore_index=True)
            
        for _ in self.sensors:
            self.temp = self.temp.append(_.temp, ignore_index=True)
            
        self.char = pd.DataFrame()
        
        self.char = pd.DataFrame()
        
        p = ["X TCO (mg/C)", "Y TCO (mg/C)", "Z TCO (mg/C)", "X TCS (%/C)", "Y TCS (%/C)", "Z TCS (%/C)"]
        
        for i in p:
            
            self.char.set_value("Average", i, self.summary[i].mean())
            self.char.set_value("St. Deviation", i, self.summary[i].std())
            self.char.set_value("Maximum", i, self.summary[i].max())
            self.char.set_value("Minimum", i, self.summary[i].min())
            self.char.set_value("Abs. Average", i, self.summary[i].abs().mean())
            
    def hist_tco(self, axis):
        
        fig = plt.figure(figsize=(3.4, 2.25))                                                            
        ax = fig.add_subplot(1,1,1) 
        
        percent_lim = .5 # will be represented as percent (* 100)
        
        xlim = 2
        xlim_maj_step = 1  # must be whole number
        xlim_min_step = 0.25   # must be whole number
        
        # paramter will be the x labels
        x_maj_ticks = [-xlim+xlim_maj_step*i for i in itertools.takewhile(lambda x : -xlim+xlim_maj_step*x <= xlim, range(int((2*xlim)/xlim_maj_step) + 1))]
        
        # parameter will also be used for bin size
        x_min_ticks = [-xlim+xlim_min_step*i for i in itertools.takewhile(lambda x : -xlim+xlim_min_step*x <= xlim, range(int((2*xlim)/xlim_min_step) + 1))]
        
        ylim = 1
        ylim_maj_step = .1  # will be represented as percent (* 100)
        ylim_min_step = .05 # will be represented as percent (* 100)
        
        y_maj_ticks = [ylim_maj_step*i for i in itertools.takewhile(lambda x : ylim_maj_step*x <= ylim, range(int(ylim/ylim_maj_step) + 1))]
        y_min_ticks = [ylim_min_step*i for i in itertools.takewhile(lambda x : ylim_min_step*x <= ylim, range(int(ylim/ylim_min_step) + 1))]
        
        hist, bins = np.histogram(self.summary["%s TCO (mg/C)" % axis.upper()], bins=x_min_ticks)
        ax.bar(bins[:-1] + 0.5 * (bins[1]-bins[0]), hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]), color='grey', edgecolor='black')

        formatter = FuncFormatter(to_percent)

        # Set the formatter
        plt.gca().yaxis.set_major_formatter(formatter)
        
        plt.xlabel('%s TCO (mg/°C)' % axis.upper())
        plt.ylabel('Percent of Population (%)')
        
        ax.set_xticks(x_maj_ticks)                                                       
        ax.set_xticks(x_min_ticks, minor=True)        
        ax.set_yticks(y_maj_ticks)                                    
        ax.set_yticks(y_min_ticks, minor=True)   

        ax.set_ylim([0, percent_lim])                                                    

        ax.grid(which='both') 
        ax.grid(which='minor', alpha=0.2)                                                
        ax.grid(which='major', alpha=0.5)  
        ax.set_axisbelow(True)
        
    def plot_offset(self, axis):
        
        fig = plt.figure(figsize=(3.4, 2.25))
        ax = fig.add_subplot(1,1,1)
        
        for sensor in tco.sensors:
            
            plt.plot(sensor.temp["Temperature"], sensor.temp["%s Offset (mg)" % axis.upper()])
            
#        ax.set_title("%s Offset v Temperature" % axis.upper())
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Offset (mg)")
        
        xlim_max = 100
        xlim_min = -100
        xlim_maj_step = 25
        xlim_min_step = 25
        
        ylim_max = 300
        ylim_min = -300
        ylim_maj_step = 100
        ylim_min_step = 50
        
        x_maj_ticks = [xlim_min+xlim_maj_step*i for i in itertools.takewhile(lambda x : xlim_min+xlim_maj_step*x <= xlim_max, range(int((2*xlim_max)/xlim_maj_step) + 1))]
        x_min_ticks = [xlim_min+xlim_min_step*i for i in itertools.takewhile(lambda x : xlim_min+xlim_min_step*x <= xlim_max, range(int((2*xlim_max)/xlim_min_step) + 1))]
        
        y_maj_ticks = [ylim_min+ylim_maj_step*i for i in itertools.takewhile(lambda x : ylim_min+ylim_maj_step*x <= ylim_max, range(int((2*ylim_max)/ylim_maj_step) + 1))]
        y_min_ticks = [ylim_min+ylim_min_step*i for i in itertools.takewhile(lambda x : ylim_min+ylim_min_step*x <= ylim_max, range(int((2*ylim_max)/ylim_min_step) + 1))]
        
        ax.set_xticks(x_maj_ticks)                                                       
        ax.set_xticks(x_min_ticks, minor=True)        
        ax.set_yticks(y_maj_ticks)                                    
        ax.set_yticks(y_min_ticks, minor=True) 
    
        ax.grid(which='both') 
        ax.grid(which='minor', alpha=0.2)                                                
        ax.grid(which='major', alpha=0.5)
        
    def hist_tcs(self, axis):
        
        
        tcs_data = pd.DataFrame()
        for n, i in enumerate(self.temp["Part ID"].unique()):
            
            part_data = self.temp[self.temp["Part ID"] == i]["%s Sensitivity (LSB/g)" % axis.upper()]
            
            part_data = part_data.reset_index()
            tcs_data.set_value(n, "%s Sensitivity (LSB/g)" % axis.upper(), part_data["%s Sensitivity (LSB/g)" % axis.upper()][0])
        
        fig = plt.figure(figsize=(9,6))                                                             
        ax = fig.add_subplot(1,1,1) 
        
        percent_lim = .3 # will be represented as percent (* 100)
        
        xlim_max = 288
        xlim_min = 224
        xlim_maj_step = 16
        xlim_min_step = 4
        
        # paramter will be the x labels
        x_maj_ticks = [xlim_min+xlim_maj_step*i for i in itertools.takewhile(lambda x : xlim_min+xlim_maj_step*x <= xlim_max, range(int((2*xlim_max)/xlim_maj_step) + 1))]
        
        # parameter will also be used for bin size
        x_min_ticks = [xlim_min+xlim_min_step*i for i in itertools.takewhile(lambda x : xlim_min+xlim_min_step*x <= xlim_max, range(int((2*xlim_max)/xlim_min_step) + 1))]
        
        ylim = 1
        ylim_maj_step = .1  # will be represented as percent (* 100)
        ylim_min_step = .05 # will be represented as percent (* 100)
        
        y_maj_ticks = [ylim_maj_step*i for i in itertools.takewhile(lambda x : ylim_maj_step*x <= ylim, range(int(ylim/ylim_maj_step) + 1))]
        y_min_ticks = [ylim_min_step*i for i in itertools.takewhile(lambda x : ylim_min_step*x <= ylim, range(int(ylim/ylim_min_step) + 1))]
        
        hist, bins = np.histogram(tcs_data["%s Sensitivity (LSB/g)" % axis.upper()], bins=x_min_ticks)
        ax.bar(bins[:-1] + 0.5 * (bins[1]-bins[0]), hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]), color='grey', edgecolor='black')

        formatter = FuncFormatter(to_percent)

        # Set the formatter
        plt.gca().yaxis.set_major_formatter(formatter)
        
        plt.xlabel('%s Sensitivity (LSB/g)' % axis.upper())
        plt.ylabel('Percent of Population (%)')
        
        ax.set_xticks(x_maj_ticks)                                                       
        ax.set_xticks(x_min_ticks, minor=True)        
        ax.set_yticks(y_maj_ticks)                                    
        ax.set_yticks(y_min_ticks, minor=True)   

        ax.set_ylim([0, percent_lim])                                                    

        ax.grid(which='both') 
        ax.grid(which='minor', alpha=0.2)                                                
        ax.grid(which='major', alpha=0.5)  
        ax.set_axisbelow(True)
        
    def hist_tcs_pcnt(self, axis):
        
        fig = plt.figure(figsize=(9,6))                                                             
        ax = fig.add_subplot(1,1,1) 
        
        percent_lim = .3 # will be represented as percent (* 100)
        
        xlim = 2
        xlim_maj_step = 0.5  # must be whole number
        xlim_min_step = 0.1   # must be whole number
        
        # paramter will be the x labels
        x_maj_ticks = [-xlim+xlim_maj_step*i for i in itertools.takewhile(lambda x : -xlim+xlim_maj_step*x <= xlim, range(int((2*xlim)/xlim_maj_step) + 1))]
        
        # parameter will also be used for bin size
        x_min_ticks = [-xlim+xlim_min_step*i for i in itertools.takewhile(lambda x : -xlim+xlim_min_step*x <= xlim, range(int((2*xlim)/xlim_min_step) + 1))]
        
        ylim = 1
        ylim_maj_step = .1  # will be represented as percent (* 100)
        ylim_min_step = .05 # will be represented as percent (* 100)
        
        y_maj_ticks = [ylim_maj_step*i for i in itertools.takewhile(lambda x : ylim_maj_step*x <= ylim, range(int(ylim/ylim_maj_step) + 1))]
        y_min_ticks = [ylim_min_step*i for i in itertools.takewhile(lambda x : ylim_min_step*x <= ylim, range(int(ylim/ylim_min_step) + 1))]
        
        hist, bins = np.histogram(self.summary["%s TCS (%%/C)" % axis.upper()], bins=x_min_ticks)
        ax.bar(bins[:-1] + 0.5 * (bins[1]-bins[0]), hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]), color='grey', edgecolor='black')

        formatter = FuncFormatter(to_percent)

        # Set the formatter
        plt.gca().yaxis.set_major_formatter(formatter)
        
        plt.xlabel('%s TCO (%%/°C)' % axis.upper())
        plt.ylabel('Percent of Population (%)')
        
        ax.set_xticks(x_maj_ticks)                                                       
        ax.set_xticks(x_min_ticks, minor=True)        
        ax.set_yticks(y_maj_ticks)                                    
        ax.set_yticks(y_min_ticks, minor=True)   

        ax.set_ylim([0, percent_lim])                                                    

        ax.grid(which='both') 
        ax.grid(which='minor', alpha=0.2)                                                
        ax.grid(which='major', alpha=0.5)  
        ax.set_axisbelow(True)
        
    def plot_dig_sens(self, axis):
        
        fig = plt.figure(figsize=(9,6))
        ax = fig.add_subplot(1,1,1)
        
        for sensor in tco.sensors:
            
            plt.plot(sensor.temp["Temperature"], sensor.temp["%s Sensitivity (LSB/g)" % axis.upper()])
            
#        ax.set_title("%s Sensitivity v Temperature" % axis.upper())
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Sensitivity (LSB/g)")
        
        xlim_max = 100
        xlim_min = -60
        xlim_maj_step = 20
        xlim_min_step = 5
        
        expected_sensitivity = 2 ** self.resolution
        
        ylim_max = expected_sensitivity + expected_sensitivity/8
        ylim_min = expected_sensitivity - expected_sensitivity/8
        ylim_maj_step = expected_sensitivity/16
        ylim_min_step = expected_sensitivity/64
        
        x_maj_ticks = [xlim_min+xlim_maj_step*i for i in itertools.takewhile(lambda x : xlim_min+xlim_maj_step*x <= xlim_max, range(int((2*xlim_max)/xlim_maj_step) + 1))]
        x_min_ticks = [xlim_min+xlim_min_step*i for i in itertools.takewhile(lambda x : xlim_min+xlim_min_step*x <= xlim_max, range(int((2*xlim_max)/xlim_min_step) + 1))]
        
        y_maj_ticks = [ylim_min+ylim_maj_step*i for i in itertools.takewhile(lambda x : ylim_min+ylim_maj_step*x <= ylim_max, range(int((2*ylim_max)/ylim_maj_step) + 1))]
        y_min_ticks = [ylim_min+ylim_min_step*i for i in itertools.takewhile(lambda x : ylim_min+ylim_min_step*x <= ylim_max, range(int((2*ylim_max)/ylim_min_step) + 1))]
        
        ax.set_xticks(x_maj_ticks)                                                       
        ax.set_xticks(x_min_ticks, minor=True)        
        ax.set_yticks(y_maj_ticks)                                    
        ax.set_yticks(y_min_ticks, minor=True) 
    
        ax.grid(which='both') 
        ax.grid(which='minor', alpha=0.2)                                                
        ax.grid(which='major', alpha=0.5)

def connect_sql():
    # TODO(Ronnie): Handle with configparser
    # connect to mysql and pull data
    
    # SJ - '10.10.40.9'
    # TW - '10.30.10.2'
    
    return pms.connect(host='10.10.40.9', port=3306, user='charuser', 
                       passwd='charuser', db='mcubeecodb')
        
def pull_data(conn, table, columns, tins):
    # collect info for query
    tins = " or ".join(["tin=%d" % i for i in tins])
    columns = ", ".join(columns)
    query = "SELECT %s FROM %s WHERE %s" % (columns, table, tins)
    return pd.read_sql_query(query, conn)

def preprocess(data):
    ##########################################################################
    # prep data - eventually should be phased out and stored this way in labview
    # convert motor angles relative to x,y,z
    # use configparser
    
    # convert to relative Z
    data.replace('[0 , 0]'      , '[+Z]'  , inplace=True)
    data.replace('[180 , -180]' , '[-Z]'  , inplace=True)
    
    # convert to relative Y
    data.replace('[-90 , -90]' , '[+Y]'  , inplace=True)
    data.replace('[90 , -90]'  , '[-Y]'  , inplace=True)
    
    # convert to relative X
    data.replace('[0 , -90]'   , '[+X]'  , inplace=True)
    data.replace('[180 , -90]' , '[-X]'  , inplace=True)
        
    return data
        
    ##########################################################################
    
def preprocess_csp(data):
    ##########################################################################
    # prep data - eventually should be phased out and stored this way in labview
    # convert motor angles relative to x,y,z
    # use configparser
    
    # convert to relative Z
    data.replace('[0 , 0]' , '[+Z]'  , inplace=True)
    data.replace('[180 , -180]'      , '[-Z]'  , inplace=True)
    
    # convert to relative Y
    data.replace('[180 , -90]' , '[+Y]'  , inplace=True)
    data.replace('[0 , -90]'  , '[-Y]'  , inplace=True)
    
    # convert to relative X
    data.replace('[-90 , -90]'   , '[+X]'  , inplace=True)
    data.replace('[90 , -90]' , '[-X]'  , inplace=True)
        
    return data
        
    ##########################################################################
    
def adj_temp_96(data):
    
    adj_data = pd.DataFrame()
    
    for board in data.BoardNumber.unique():
        
        curr_pos = 0
        
        temperatures = [25, 55, 85, -20, -40, 25]
        
        for temp in temperatures:
                
            temp_data = data[data.BoardNumber == board][curr_pos:curr_pos + 96]
            temp_data.BoardTemp = temp
            adj_data = adj_data.append(temp_data)
            
            curr_pos += 96
    
    return adj_data

def adj_temp_16(data):
    
    adj_data = pd.DataFrame()
    
    for board in data.BoardNumber.unique():
        
        curr_pos = 0
        
        temperatures = [25, 55, 85, -20, -40, 25]
        
        for temp in temperatures:
                
            temp_data = data[data.BoardNumber == board][curr_pos:curr_pos + 16]
            temp_data.BoardTemp = temp
            adj_data = adj_data.append(temp_data)
            
            curr_pos += 16
    
    return adj_data

#%%
    
if __name__ == "__main__":
    conn = connect_sql()
    
    tins = [17401]
    
    data = pull_data(conn, 'lit', '*', tins)
    adj_data = preprocess_csp(data)
    adj_data = adj_temp_96(adj_data)
    tco = TCO(adj_data)
    outdir = "C:/Users/"+getpass.getuser()+"/Desktop/PythonAutomatedProcessing/tco/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)    
    
    tco.char.to_csv(outdir+"tco.csv")
    tco.summary.to_csv(outdir+"tco_sum.csv")
    
    tco.hist_tco('x')
    plt.savefig(outdir+"x_tco.png", bbox_inches='tight')
    
    tco.hist_tco('y')
    plt.savefig(outdir+"y_tco.png", bbox_inches='tight')
    
    tco.hist_tco('z')
    plt.savefig(outdir+"z_tco.png", bbox_inches='tight')
    
    tco.plot_offset('x')
    plt.savefig(outdir+"x_offset.png", bbox_inches='tight')
    
    tco.plot_offset('y')
    plt.savefig(outdir+"y_offset.png", bbox_inches='tight')
    
    tco.plot_offset('z')
    plt.savefig(outdir+"z_offset.png", bbox_inches='tight')
    
    
#    tco.hist_tcs('x')
#    tco.hist_tcs('y')
#    tco.hist_tcs('z')