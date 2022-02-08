# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 15:14:52 2017

@author: ronnie
"""

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pymysql as pms

sensor_data_cols = ('partId', 'Ts', 'Vs', 'Vm', 'rbm_x', 'rbm_y', 'rbm_z', 'rbm_x_off', 'rbm_y_off', 'rbm_z_off', 'rbm_x_off_mg', 'rbm_y_off_mg', 'rbm_z_off_mg')
sensor_sum_cols = ('partId', 'wafer', 'lot', 'y_loc', 'x_loc', 'Vs', 'Vm', 'x_tco', 'y_tco', 'z_tco')

class sensor(object):
    
    mems_sens = 0.00225
    
    def __init__(self, df, x_sens, y_sens, z_sens):
        
        self.data = pd.DataFrame(columns=sensor_data_cols)
        self.summary = pd.DataFrame(columns=sensor_sum_cols)
        
        self.df = df
        self.x_sens = x_sens
        self.y_sens = y_sens
        self.z_sens = z_sens
        
        df = df[df.DVs == 1.7]
        df = df[df.DVm == 0]
        
        for n, i in enumerate(df.idtco.unique()):
            
            curr_data = df[df.idtco == i]
           
            self.data.loc[n, 'partId'] = int(curr_data.partId.item())
            self.data.loc[n, 'Ts'] = int(curr_data.Ts.item())
            self.data.loc[n, 'Vs'] = curr_data.DVs.item()
            self.data.loc[n, 'Vm'] = curr_data.DVm.item()
            self.data.loc[n, 'rbm_x'] = curr_data.X.item()
            self.data.loc[n, 'rbm_y'] = curr_data.Y.item()
            self.data.loc[n, 'rbm_z'] = curr_data.Z.item()
            
            if n == 0:
                self.base_rbm_x = self.data.rbm_x.item()
                self.base_rbm_y = self.data.rbm_y.item()
                self.base_rbm_z = self.data.rbm_z.item()
                
            self.data.loc[n, 'rbm_x_off'] = self.data.rbm_x[n] - self.base_rbm_x
            self.data.loc[n, 'rbm_y_off'] = self.data.rbm_y[n] - self.base_rbm_y
            self.data.loc[n, 'rbm_z_off'] = self.data.rbm_z[n] - self.base_rbm_z
            self.data.loc[n, 'rbm_x_off_mg'] = 1000* (self.mems_sens/self.x_sens) * (self.data.rbm_x[n] - self.base_rbm_x)
            self.data.loc[n, 'rbm_y_off_mg'] = 1000* (self.mems_sens/self.y_sens) * (self.data.rbm_y[n] - self.base_rbm_y)
            self.data.loc[n, 'rbm_z_off_mg'] = 1000* (self.mems_sens/self.z_sens) * (self.data.rbm_z[n] - self.base_rbm_z)
                         
        high_temp = self.data.Ts.max()
        low_temp = self.data.Ts.min()
        
        x_span = self.data[self.data.Ts == high_temp].rbm_x_off.item() - self.data[self.data.Ts == low_temp].rbm_x_off.item()
        y_span = self.data[self.data.Ts == high_temp].rbm_y_off.item() - self.data[self.data.Ts == low_temp].rbm_y_off.item()
        z_span = self.data[self.data.Ts == high_temp].rbm_z_off.item() - self.data[self.data.Ts == low_temp].rbm_z_off.item()
        
        x_rbm_tco = x_span / (high_temp - low_temp)
        y_rbm_tco = y_span / (high_temp - low_temp)
        z_rbm_tco = z_span / (high_temp - low_temp)
        
        
        
        self.summary.loc[0, 'partId'] = int(self.data.partId.unique().item())
        self.summary.loc[0, 'wafer'] =   (int(self.data.partId.unique().item()) & 0xff000000) >> 24
        self.summary.loc[0, 'lot'] = (int(self.data.partId.unique().item()) & 0x00ff0000) >> 16
        self.summary.loc[0, 'y_loc'] = (int(self.data.partId.unique().item()) & 0x0000ff00) >> 8
        self.summary.loc[0, 'x_loc'] = (int(self.data.partId.unique().item()) & 0x000000ff)
        self.summary.loc[0, 'Vs'] = int(self.data.Vs.unique().item())
        self.summary.loc[0, 'Vm'] = int(self.data.Vm.unique().item())
        self.summary.loc[0, 'x_tco'] = (1000 * (self.mems_sens/self.x_sens) * x_rbm_tco)
        self.summary.loc[0, 'y_tco'] = (1000 * (self.mems_sens/self.y_sens) * y_rbm_tco)
        self.summary.loc[0, 'z_tco'] = (1000 * (self.mems_sens/self.z_sens) * z_rbm_tco)
    
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

class socket_tco(object):
    
    path = os.getcwd() + '\\tco_automation_%s\\' % datetime.now().strftime("%y%m%d%H%M")
    file_type = '.png'
    
    def __init__(self, tins):
        
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        
        # setup style for matplotlib
        plt.style.use('ggplot')
        
        if tins is None:
            raise ValueError('No TINs passed in')
        
        # collect info for query
        tins = " or ".join(["tin=%d" % i for i in tins])
        query = "SELECT * FROM tco WHERE %s" % tins

        # TODO(Ronnie): Handle with configparser
        # connect to mysql and pull data
        self.conn = pms.connect(host='10.10.40.9', port=3306, user='charuser', 
                                passwd='charuser', db='mcubeecodb')
        self.df = pd.read_sql_query(query, self.conn)
        
        # add mems_sens as columns for each device
        sens_df = self._pull_mems_sens()
        
        self.sensors = []
        
        for _ in self.df.partId.unique():
            
            self.sensors.append(sensor(self.df[self.df.partId == _],
                                       sens_df[sens_df.partId == _].Xsens.item(),
                                       sens_df[sens_df.partId == _].Ysens.item(),
                                       sens_df[sens_df.partId == _].Zsens.item()))
            
        self.data = pd.DataFrame(columns=sensor_data_cols)
        self.summary = pd.DataFrame(columns=sensor_sum_cols)
        
        for _ in self.sensors:
            self.data = self.data.append(_.data, ignore_index=True)
            self.summary = self.summary.append(_.summary, ignore_index=True)
            
        self.find_electrical_failures()
    
    def _pull_mems_sens(self):
        
        if self.df.empty:
            return
        
        part_ids = " or ".join(["partId=%d" % i for i in self.df.partId.unique()])
        query = "SELECT * FROM mems_sens WHERE %s" % part_ids
        
        mems_sens = pd.read_sql_query(query, self.conn)
        
        return mems_sens
    
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, value):
        self._data = value
        
    def find_electrical_failures(self, pull_from_summary=True):
        self.fail_data = pd.DataFrame(columns=sensor_data_cols)
        self.fail_sum = pd.DataFrame(columns=sensor_sum_cols)
        
        self.fail_data = self.data[((self.data.rbm_x == 0) &
                                    (self.data.rbm_y == 0) & 
                                    (self.data.rbm_z == 0)) | 
                                   ((self.data.rbm_x == -256) &
                                    (self.data.rbm_y == -1) & 
                                    (self.data.rbm_z == -1))]
        
        if pull_from_summary:
            
            for i in self.fail_data.partId.unique():
                
                self.summary = self.summary[self.summary.partId != i]
                self.fail_sum = self.fail_sum.append(self.summary[self.summary.partId == i])
    
    def offset_v_temp_plot(self, save=True):
        
        if save:
                plt.ioff()
        
        for axis in ['x', 'y', 'z']:
            
            plt.figure(figsize=(12,7.5))
            
            file_title = '%s - Offset v Temperature' % axis.upper()
            plot_title = file_title
            ylabel = '%s Offset (mg)' % axis.upper()
        
            for i in self.summary.partId.unique():
                
                curr_dut = self.data[self.data.partId == i]
                
                data_col = 'rbm_%s_off_mg' % axis
                
                plt.plot(curr_dut.Ts, curr_dut[data_col], marker='o')
                
            ax = plt.gca()
            ax.set_ylim([-300, 300])
            ax.set_title(plot_title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel('Temperature (C)')
            
            if save:
                plt.savefig(self.path + '\\' + file_title + self.file_type, bbox_inches='tight', dpi=300)
                plt.close()
            else:
                plt.show()
            
    
    def wafer_bubble_plot(self):
        pass
    
if __name__ == '__main__':    
    tins = [13308,13309,13295,13303,13304,13305]
#    tins = [20422,20423]
    test = socket_tco(tins)
    test.offset_v_temp_plot()