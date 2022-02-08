# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 16:02:09 2017

@author: ronnie

Modified by Tyler Walker on Fri Mar 6 14:17:09 2020
"""

import configparser as cp
from datetime import datetime
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
import pymysql as pms
import pandas as pd

drop_cols = ('lot', 'wafer', 'part_id', 'height', 'drop', 'dig_x', 'dig_y', 'dig_z', 'rbm_x', 'rbm_y', 'rbm_z', 'failure')
sensor_cols = ('lot', 'wafer', 'part_id', 'loc_x', 'loc_y', 'bd_loc', 'bd_num', 'x_sens', 'y_sens', 'z_sens', 'fail_tot', 'fail_uni', 'fail_30', 'fail_45', 'fail_60')
wafer_cols = ('lot', 'wafer', 'duts', 'fail_tot', 'fail_uni', 'fail_30', 'fail_45', 'fail_60')
lot_cols = ('lot', 'fail_tot', 'fail_uni', 'fail_30', 'fail_45', 'fail_60')
board_cols = ('board', 'duts', 'fail_tot', 'fail_uni', 'fail_30', 'fail_45', 'fail_60', 'loc1', 'loc2', 'loc3', 'loc4', 'loc5', 'loc6', 'loc7', 'loc8', 'loc9', 'loc10', 'loc11', 'loc12', 'loc13', 'loc14', 'loc15', 'loc16', 'loc17', 'loc18', 'loc19', 'loc20', 'loc21', 'loc22', 'loc23', 'loc24', 'loc25', 'loc26', 'loc27', 'loc28', 'loc29', 'loc30', 'loc31', 'loc32')

def rand_jitter(arr):
        stdev = max(arr)/15.
        return arr + np.random.randn(len(arr)) * stdev

class drop(object):
    """
    """
    
    #TODO(): Eventually set up thru configparser
    limit_g = 0.350
    
    def __init__(self, df, resolution):
        self.summary = pd.DataFrame(columns=drop_cols)
        
        self.resolution = resolution
        
        self.summary.loc[0, 'lot']      = str(df.LotID.unique()[0])
        self.summary.loc[0, 'wafer']    = int(df.WaferNumber.unique())
        self.summary.loc[0, 'part_id']  = int(df.ChipID.unique())
        self.summary.loc[0, 'height']  = str(df.Height.to_string(index=False))
        self.summary.loc[0, 'drop']    = int(df.CurrentDrop)
        self.summary.loc[0, 'dig_x']   = int(df.X_digital)
        self.summary.loc[0, 'dig_y']   = int(df.Y_digital)
        self.summary.loc[0, 'dig_z']   = int(df.Z_digital)
        self.summary.loc[0, 'rbm_x']   = int(df.X_RBM)
        self.summary.loc[0, 'rbm_y']   = int(df.Y_RBM)
        self.summary.loc[0, 'rbm_z']   = int(df.Z_RBM)
        self.summary.loc[0, 'failure'] = self.check_failure()
    
    @property
    def summary(self):
        return self._summary
    
    @summary.setter
    def summary(self, value):
        self._summary = value
        
    def check_failure(self):
        
        if ((float(np.abs(int(self.summary.dig_x) - 0)) / 2**self.resolution) > self.limit_g):
            return 1
        
        if ((float(np.abs(int(self.summary.dig_y) - 0)) / 2**self.resolution) > self.limit_g):
            return 1
        
        if ((float(np.abs(int(self.summary.dig_z) - 2**self.resolution)) / 2**self.resolution) > self.limit_g):
            return 1
    
        return 0
    
class sensor(object):
    """
    """
    
    def __init__(self, df, resolution):
        self.df = df
        self.data = pd.DataFrame(columns=drop_cols)
        self.summary = pd.DataFrame(columns=sensor_cols)
        self.fails = pd.DataFrame(columns=drop_cols)
        
        self.summary.loc[0, 'lot']      = str(df.LotID.unique()[0])
        self.summary.loc[0, 'wafer']    = int(df.WaferNumber.unique()[0])
        self.summary.loc[0, 'part_id']  = int(df.ChipID.unique()[0])
        self.summary.loc[0, 'loc_x']    = int(df.X_location.unique()[0])
        self.summary.loc[0, 'loc_y']    = int(df.Y_Location.unique()[0])
        self.summary.loc[0, 'bd_loc']   = int(df.Site.unique()[0])
        self.summary.loc[0, 'bd_num']   = int(df.BoardNumber.unique()[0])
        
        # calculate sensitivity
        initial = df[df.Height == 'Initial']
        
        try:
            initial_xp = int(initial[initial.LastKnownOrient == '[+X]']['X_RBM'].iloc[-1])
        except:
            print('BD %d - DUT %d - [+X] initial reading not detected.' % 
                  (self.summary.bd_num.item(), self.summary.part_id.item()))
            initial_xp = 0
            
        try:
            initial_xn = int(initial[initial.LastKnownOrient == '[-X]']['X_RBM'].iloc[-1])
        except:
            print('BD %d - DUT %d - [-X] initial reading not detected.' % 
                  (self.summary.bd_num.item(), self.summary.part_id.item()))
            initial_xn = 0
            
        try:
            initial_yp = int(initial[initial.LastKnownOrient == '[+Y]']['Y_RBM'].iloc[-1])
        except:
            print('BD %d - DUT %d - [+Y] initial reading not detected.' % 
                  (self.summary.bd_num.item(), self.summary.part_id.item()))
            initial_yp = 0
            
        try:
            initial_yn = int(initial[initial.LastKnownOrient == '[-Y]']['Y_RBM'].iloc[-1])
        except:
            print('BD %d - DUT %d - [-Y] initial reading not detected.' % 
                  (self.summary.bd_num.item(), self.summary.part_id.item()))
            initial_yn = 0
            
        try:
            initial_zp = int(initial[initial.LastKnownOrient == '[+Z]']['Z_RBM'].iloc[-1])
        except:
            print('BD %d - DUT %d - [+Z] initial reading not detected.' % 
                  (self.summary.bd_num.item(), self.summary.part_id.item()))
            initial_zp = 0
            
        try:
            initial_zn = int(initial[initial.LastKnownOrient == '[-Z]']['Z_RBM'].iloc[-1])
        except:
            print('BD %d - DUT %d - [-Z] initial reading not detected.' % 
                  (self.summary.bd_num.item(), self.summary.part_id.item()))
            initial_zn = 0
            
        self.summary.loc[0, 'x_sens']   = np.abs(initial_xp - initial_xn) / 2 / (((2**15)/48) / 1.536)
        self.summary.loc[0, 'y_sens']   = np.abs(initial_yp - initial_yn) / 2 / (((2**15)/48) / 1.536)
        self.summary.loc[0, 'z_sens']   = np.abs(initial_zp - initial_zn) / 2 / (((2**15)/48) / 1.536)
        
        drop_df = df[(df.TestItem == 'Towel') &
                     ((df.Height == '45cm') | 
                     (df.Height == '60cm') | 
                     (df.Height == '30cm'))]
        
        # create drop objects
        self.drops = []
        for _ in drop_df.idLIT:
            self.drops.append(drop(drop_df[drop_df.idLIT == _], resolution))
            
        # compile data from drops
        for _ in self.drops:
            self.data = self.data.append(_.summary, ignore_index=True)
            
        # pull data from failed drops
        self.fails = self.data[self.data.failure == 1]
        
        # count failures
        self.summary.fail_tot = 0
        self.summary.fail_uni = 0
        self.summary.fail_30 = 0
        self.summary.fail_45 = 0
        self.summary.fail_60 = 0
            
        prev_fail = False
        dev_fail = False
        for _ in self.drops:
            
            if _.summary.failure.item() == 1:
                
                if not dev_fail:
                    self.summary.fail_tot = self.summary.fail_tot + 1
                    dev_fail = True
                
                if not prev_fail:
                    self.summary.fail_uni = self.summary.fail_uni + 1
                    
                    if _.summary.height.to_string(index=False) == '30cm':
                        self.summary.fail_30 = self.summary.fail_30 + 1
                    
                    if _.summary.height.to_string(index=False) == '45cm':
                        self.summary.fail_45 = self.summary.fail_45 + 1
                    
                    if _.summary.height.to_string(index=False) == '60cm':
                        self.summary.fail_60 = self.summary.fail_60 + 1
                      
                
                prev_fail = True
                        
            else:
                prev_fail = False
                    
    
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
        
    @property
    def fails(self):
        return self._fails
    
    @fails.setter
    def fails(self, value):
        self._fails = value
    
class wafer(object):
    """
    """
    
    def __init__(self, df):
        self.data = pd.DataFrame(columns=sensor_cols)
        self.summary = pd.DataFrame(columns=wafer_cols)
        self.fails = pd.DataFrame(columns=drop_cols)

        self.summary.loc[0, 'lot']   = str(df.LotID.unique()[0])
        self.summary.loc[0, 'wafer'] = int(df.WaferNumber.unique())
        
        print('Gathering data from %s W%s' % (self.summary.lot.item(), 
                                              self.summary.wafer.item()))
        
        self._name = str(self.summary.wafer.item())
        
        drop_df = df[(df.TestItem == 'Towel') &
                     ((df.Height == '45cm') | 
                     (df.Height == '60cm') | 
                     (df.Height == '30cm'))]
        
        # detect resolution
        try:
            self.resolution = int((np.log(drop_df[drop_df.LastKnownOrient == '[+Z]']['Z_digital'].mean())/np.log(2)) + 0.5)
        except:
            print(drop_df[drop_df.LastKnownOrient == '[+Z]']['Z_digital'].mean())
            self.resolution = 2
        
        self.sensors = []
        for _ in df.ChipID.unique():
            self.sensors.append(sensor(df[df.ChipID == _], self.resolution))
            
        # compile data from sensors
        for _ in self.sensors:
            self.data = self.data.append(_.summary, ignore_index=True)
            self.fails = self.fails.append(_.fails, ignore_index=True)
            
        self.summary.loc[0, 'duts'] = len(self.sensors)
            
        # count failures
        self.summary.fail_tot = 0
        self.summary.fail_uni = 0
        self.summary.fail_30 = 0
        self.summary.fail_45 = 0
        self.summary.fail_60 = 0
        
        for _ in self.sensors:
            if _.summary.fail_tot[0] > 0:
                self.summary.fail_tot = self.summary.fail_tot + _.summary.fail_tot
                self.summary.fail_uni = self.summary.fail_uni + _.summary.fail_uni
                self.summary.fail_30  = self.summary.fail_30  + _.summary.fail_30
                self.summary.fail_45  = self.summary.fail_45  + _.summary.fail_45
                self.summary.fail_60  = self.summary.fail_60  + _.summary.fail_60
                
        
                
    @property
    def name(self):
        return self._name
    
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
        
    @property
    def fails(self):
        return self._fails
    
    @fails.setter
    def fails(self, value):
        self._fails = value
        
class lot(object):
    """
    """
    
    def __init__(self, df):
        self.data = pd.DataFrame(columns=wafer_cols)
        self.summary = pd.DataFrame(columns=lot_cols)
        self.fails = pd.DataFrame(columns=drop_cols)

        self.summary.loc[0, 'lot'] = df.LotID.unique().item()
        self._name = self.summary.lot.item()
        
        self.wafers = []
        for _ in df.WaferNumber.unique():
            
            # wafer == 0, means electrical fail
            if _ != 0:
                self.wafers.append(wafer(df[df.WaferNumber == _]))
            
        # compile data from wafers
        for _ in self.wafers:
            self.data = self.data.append(_.summary, ignore_index=True)
            self.fails = self.fails.append(_.fails, ignore_index=True)
            
        # count failures
        self.summary.fail_tot = 0
        self.summary.fail_uni = 0
        self.summary.fail_30 = 0
        self.summary.fail_45 = 0
        self.summary.fail_60 = 0
        
        for _ in self.wafers:
            if _.summary.fail_tot[0] > 0:
                self.summary.fail_tot = self.summary.fail_tot + _.summary.fail_tot
                self.summary.fail_uni = self.summary.fail_uni + _.summary.fail_uni
                self.summary.fail_30  = self.summary.fail_30  + _.summary.fail_30
                self.summary.fail_45  = self.summary.fail_45  + _.summary.fail_45
                self.summary.fail_60  = self.summary.fail_60  + _.summary.fail_60
                
        self.wafers.sort(key=lambda x: x.name, reverse=False)
                
    @property
    def name(self):
        return self._name
    
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
        
    @property
    def fails(self):
        return self._fails
    
    @fails.setter
    def fails(self, value):
        self._fails = value
        
class board(object):
    """
    """
    
    def __init__(self, df):
        self.data = pd.DataFrame(columns=sensor_cols)
        self.data = self.data.append(df, ignore_index=True)
        self.data.sort_values(['lot','wafer'], inplace=True)

        # gather summary info        
        self.summary = pd.DataFrame(columns=board_cols)
        self.summary.loc[0, 'board']      = self.data.bd_num.unique().item()
        self.summary.loc[0, 'duts']       = len(self.data.index)
        self.summary.loc[0, 'fail_tot']   = self.data.fail_tot.sum()
        self.summary.loc[0, 'fail_uni']   = self.data.fail_uni.sum()
        self.summary.loc[0, 'fail_30']    = self.data.fail_30.sum()
        self.summary.loc[0, 'fail_45']    = self.data.fail_45.sum()
        self.summary.loc[0, 'fail_60']    = self.data.fail_60.sum()
        
        # initialize location failures to 0
        for _ in range(1,33):
            curr_loc = 'loc%d' % _
            self.summary.loc[0, curr_loc] = 0
            
        for _ in self.data.bd_loc.unique():
            curr_loc = 'loc%d' % _
            self.summary.loc[0, curr_loc] = self.data[self.data.bd_loc == _].fail_tot.item()
        
        self._name = self.summary.board.item()
        
    @property
    def name(self):
        return self._name
        
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
        
    @property
    def fails(self):
        return self._fails
    
    @fails.setter
    def fails(self, value):
        self._fails = value
    
class LIT(object):
    """
    """
    
    path = os.getcwd() + '\\lit_automation_%s\\' % datetime.now().strftime("%y%m%d%H%M")
    board_path = path + 'boards' + '\\'
    lot_paths = dict()

    
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
        query = "SELECT * FROM lit WHERE %s" % tins

        # TODO(Ronnie): Handle with configparser
        # connect to mysql and pull data
        
        # SJ - '10.10.40.9'
        # TW - '10.30.10.2'
        
        self.conn = pms.connect(host='10.10.40.9', port=3306, user='charuser', 
                                passwd='charuser', db='mcubeecodb')
                                
        print('Pulling data from database...')
        self.df = pd.read_sql_query(query, self.conn)
        
        # create lots
        self.lots = []
        for _ in self.df.LotID.unique():
            
            # create folder for lot
            lot_path = self.path + _.upper() + '\\'
            self.lot_paths[_] = lot_path
            
            # self.lots.append(lot(self.df[self.df.LotID == _]))
            
            self.lots.append(lot(self.df[(self.df.LotID == _) & 
                                         (self.df.TestItem != 'Table')]))
            
        self.lots.sort(key=lambda x: x.name, reverse=False)
            
        if True:
            # create folder for boards
            os.makedirs(self.board_path)
                
            # separate data into boards
            self.boards = []
            for i in self.df.BoardNumber.unique():
        
                bd_data = pd.DataFrame(columns=sensor_cols)
        
                for j in self.lots:
                    for k in j.wafers:
                        bd_data = bd_data.append(k.data[k.data.bd_num == int(i)])
                    
                self.boards.append(board(bd_data))
            
        
            self.boards.sort(key=lambda x: x.name, reverse=False)
            
    def summaries(self):
        
        wafer_summaries = pd.DataFrame(columns=wafer_cols)
        board_summaries = pd.DataFrame(columns=board_cols)
        fail_summaries  = pd.DataFrame(columns=drop_cols)
        lot_summary     = pd.DataFrame(columns=sensor_cols)
        
        wafer_title = '%s\\%s' % (self.path, 'all_wafer_summary.csv')
        board_title = '%s\\%s' % (self.path, 'board_summary.csv')
        fail_title  = '%s\\%s' % (self.path, 'fail_summary.csv')
        
        for i in self.lots:
            fail_summaries = fail_summaries.append(i.fails, ignore_index=True)
    
        fail_summaries.to_csv(fail_title, index=False)
        
        for i in self.lots:
            
            lot_title = '%s\\%s%s' % (self.lot_paths[i.name], i.name,'_data.csv')
            
            for j in i.wafers:
                wafer_summaries = wafer_summaries.append(j.summary, 
                                                         ignore_index=True)
                
                lot_summary = lot_summary.append(j.data, ignore_index=True)
                
            lot_summary.to_csv(lot_title, index=False)
    
        wafer_summaries.to_csv(wafer_title, index=False)

        if False:
            for i in self.boards:
                board_summaries = board_summaries.append(i.summary, 
                                                         ignore_index=True)
        
            board_summaries.to_csv(board_title, index=False)
    
    def wafer_plot(self, save=True):
        
        wafer_rad = 90
        rfl_x = 20
        rfl_y = 18
        
        for i in self.lots:
            lot = i.name
            
            if not os.path.exists(self.lot_paths[lot]):
                os.makedirs(self.lot_paths[lot])
        
            for j in i.wafers:
                
                if save:
                    plt.ioff()
                
                plt.figure(figsize=(12,7.5))
                ax = plt.gca()
                
                wafer = j.name
                
                plot_title = '%s.%s - P/F Location' % (lot, wafer)
                file_title = '%s.%s_wafer_plot' % (lot, wafer)
                
                # setup wafer circle and rfl grid
                circle = patches.Circle((wafer_rad, wafer_rad), wafer_rad, 
                                        facecolor='none', 
                                        edgecolor=(0, 0.8, 0.8), linewidth=3, 
                                        alpha=0.5)
                ax.add_patch(circle)
                
                for name, group in j.data.groupby((j.data.fail_tot>0),sort=False):
                    
                    # plot passes
                    if name == False:
                        plt.plot(group.loc_x, group.loc_y, marker='o', color='g', 
                     markerfacecolor='none', linestyle='None', label='Pass')
                        
                    # plot failures
                    else:
                        plt.plot(group.loc_x, group.loc_y, marker='x', color='r', 
                                 linestyle='None', label='Fail')
                    
                ax = plt.gca()
                ax.set_ylim([-5, 2*wafer_rad+5])
                ax.set_xlim([-5, 2*wafer_rad+5])
                ax.set_aspect('equal')
                plt.yticks([i*rfl_y for i in range(int(wafer_rad*2/rfl_y + 1))])
                plt.xticks([i*rfl_x for i in range(int(wafer_rad*2/rfl_x + 1))])
                ax.invert_yaxis()
                ax.set_title(plot_title)
                ax.set_xlabel('X Location')
                ax.set_ylabel('Y Location')
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                
                if save:
                    plt.savefig(self.lot_paths[lot] + '\\' + file_title.upper() + self.file_type, bbox_inches='tight')
                    plt.close()
                else:
                    plt.show()
    
    def board_plot(self, save=True):
    
        fail_max = 5
        
        if save:
            plt.ioff()
            
        if not os.path.exists(self.board_path):
            os.makedirs(self.board_path)
        
        # colors for wafer identification
        waf_colors = {}
        color=iter(plt.cm.Dark2(np.linspace(0,1,16)))
        
        for i in self.boards:
        
            # get number of duts on board
            # only 16 and 32 dut boards exist for now
            if i.summary.duts.item() > 16:
                lit_rows = 4
                lit_cols = 8
                duts = 32
                fig = plt.figure(figsize=(22,7.5))
            else:
                lit_rows = 4 
                lit_cols = 4
                duts = 16
                fig = plt.figure(figsize=(12,7.5))
            
            ax = fig.add_subplot(111)
            
            plot_title = 'LIT Board #%d Failures' % i.data.bd_num.unique()[0]
            file_title = 'lit_bd%d' % i.data.bd_num.unique()[0]
            
            # create plot with location markers
            for x in range(lit_cols):
                for y in range(lit_rows):
                    p = patches.Rectangle((x,y),1,1,fill=False)
                    ax.add_patch(p)
                    
                    location = y*4+x+1
                    
                    # adjustment for location's weird numbering
                    if duts == 32:
                        if x > 3:
                            location = location + 12

                    x_text = x + 0.5
                    y_text = y + 0.5
                    
                    ax.text(x_text, y_text, str(location), 
                            verticalalignment='center', 
                            horizontalalignment='center')
                    
            board_fails = np.empty([lit_rows+1, lit_cols+1])
            board_fails[lit_rows][lit_cols] = fail_max
            
            for x in range(lit_cols):
                for y in range(lit_rows):
                    location = y*4+x+1
                    
                    
                    # adjustment for location's weird numbering
                    if duts == 32:
                        if x > 3:
                            location = location + 12
                    
                    curr_dut = i.data[i.data.bd_loc == location]
                    
                    if not curr_dut.empty:
                    
                        lot_wafer = '%s.%s' % (curr_dut.lot.item(), 
                                               curr_dut.wafer.item())
                        
                        if not lot_wafer in waf_colors:
                            next(color)
                            waf_colors[lot_wafer] = next(color)
                        
                        board_fails[y][x] = int(curr_dut.fail_uni.item())
                
                        p = patches.Polygon([(x+1, y+0.25),
                                             (x+1, y),
                                             (x+0.75, y)], 
                                            fill=True, 
                                            label=lot_wafer,
                                            color=waf_colors[lot_wafer])

                    else:
                    
                        board_fails[y][x] = 0 
                        p = patches.Polygon([(x+1, y+0.25),
                                             (x+1, y),
                                             (x+0.75, y)], 
                                            fill=True, 
                                            label='Unknown',
                                            color='k')
                                            
                    ax.add_patch(p)
                        
            
            # define the colormap
            cmap = plt.cm.YlOrRd
            # extract all colors from the .jet map
            cmaplist = [cmap(i) for i in range(cmap.N)]
            # force the first color entry to be grey
            cmaplist[0] = (.5,.5,.5,1.0)
            # create the new map
            cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
            
            plt.pcolor(board_fails, cmap='YlOrRd')

            cbar = plt.colorbar(ticks=range(fail_max+1), 
                                label='Number of Fails')
            
            ticks = [str(i) for i in range(fail_max+1)]
            ticks[fail_max] = '%d+' % fail_max
            cbar.ax.set_yticklabels(ticks)
            
            ax.set_xlim([0, lit_cols])
            ax.set_ylim([0, lit_rows])
            ax.set_title(plot_title)
            ax.set_aspect('equal')
            plt.xticks([])
            plt.yticks([])
            
            # sort and remove duplicate handles
            handles, labels = plt.gca().get_legend_handles_labels()
            newLabels, newHandles = [], []
            labels, handles = zip(*sorted(zip(labels, handles), 
                                          key=lambda t: t[0]))
            for handle, label in zip(handles, labels):
              if label not in newLabels:
                newLabels.append(label)
                newHandles.append(handle)
                
            # place legend outside of plot
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])            
            ax.legend(newHandles, newLabels, loc='center left', 
                      bbox_to_anchor=(1, 0.5))

            if save:
                plt.savefig(self.board_path + '\\' + file_title.upper() + self.file_type, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
    
    def sens_plot(self, save=True):
        
        if save:
                plt.ioff()
              
        for i in self.lots:
            lot = i.name
            
            if not os.path.exists(self.lot_paths[lot]):
                os.makedirs(self.lot_paths[lot])
    
            wafer_list = []
            all_wafer_data = pd.DataFrame(columns=sensor_cols)
            
            for j in i.wafers:
                
                wafer = j.name
                wafer_list.append(wafer)
                
                all_wafer_data = all_wafer_data.append(j.data, 
                                                       ignore_index=True)
                
                for k in range(3):
                    
                    plt.figure(figsize=(12,7.5))
                    ax = plt.gca()                
    
                    for name, group in j.data.groupby((j.data.fail_tot>0), 
                                                      sort=False):
                        
                        if k == 0:
                            plot_data = group.x_sens
                            plot_axis = 'X'
                        elif k == 1:
                            plot_data = group.y_sens
                            plot_axis = 'Y'
                        else:
                            plot_data = group.z_sens
                            plot_axis = 'Z'
                            
                        plot_title = '%s.%s - %s Sensitivity' % (lot, wafer, plot_axis)
                        file_title = '%s.%s_%ssens_plot' % (lot, wafer, plot_axis)
                        ylabel = '%s Sensitivity (fF/g)' % plot_axis
                        
                        # plot passes
                        if name == False:
                            plt.plot(rand_jitter([1]*len(plot_data)), plot_data, marker='o', 
                                     color='g', markerfacecolor='none', 
                                     linestyle='None', label='Pass')
                            
                        # plot failures
                        else:
                            plt.plot(rand_jitter([2]*len(plot_data)), plot_data, 
                                     marker='x', color='r', linestyle='None', 
                                     label='Fail')
                            
                    ax.set_ylim([0.8, 2.4])
                    ax.set_xlim([0.5, 2.5])
                    ax.set_title(plot_title)
                    ax.set_ylabel(ylabel)
                    box = ax.get_position()
                    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    plt.xticks([1, 2], ['Pass', 'Fail'])
                    
                    if save:
                        plt.savefig(self.lot_paths[lot] + '\\' + file_title.upper() + self.file_type, bbox_inches='tight')
                        plt.close()
                    else:
                        plt.show()
                    
            # plot all wafers in lot
            for k in range(3):
                
                wafer = ", ".join(wafer_list)
                file_wafer = "_".join(wafer_list)
            
                plt.figure(figsize=(12,7.5))
                ax = plt.gca()                
                
                j = all_wafer_data
                
                for name, group in j.groupby((j.fail_tot>0),sort=False):
                    
                    if k == 0:
                        plot_data = group.x_sens
                        plot_axis = 'X'
                    elif k == 1:
                        plot_data = group.y_sens
                        plot_axis = 'Y'
                    else:
                        plot_data = group.z_sens
                        plot_axis = 'Z'
                        
                    plot_title = '%s.%s - %s Sensitivity' % (lot, wafer, 
                                                             plot_axis)
                    file_title = '%s.%s_%ssens_plot' % (lot, file_wafer, 
                                                        plot_axis)
                    ylabel = '%s Sensitivity (fF/g)' % plot_axis
                    
                    # plot passes
                    if name == False:
                        plt.plot(rand_jitter([1]*len(plot_data)), plot_data, 
                                 marker='o', color='g', markerfacecolor='none', 
                                 linestyle='None', label='Pass')
                        
                    # plot failures
                    else:
                        plt.plot(rand_jitter([2]*len(plot_data)), plot_data, 
                                 marker='x', color='r', linestyle='None', 
                                 label='Fail')
                        
                ax.set_ylim([0.8, 2.4])
                ax.set_xlim([0.5, 2.5])
                ax.set_title(plot_title)
                ax.set_ylabel(ylabel)
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.xticks([1, 2], ['Pass', 'Fail'])
                
                if save:
                    plt.savefig(self.lot_paths[lot] + '\\' + file_title.upper() + self.file_type, bbox_inches='tight')
                    plt.close()
                else:
                    plt.show()
    
    def rbm_plot(self, save=True):
        
        if save:
                plt.ioff()
                
        rbm_fail = pd.DataFrame(columns=drop_cols)
        rbm_devs = []
        
        for i in self.lots:
            
            lot = i.name
        
            for j in i.wafers:
                
                wafer = j.name
                
                for k in j.sensors:
                        
                    fail_num = 0
                    fail_prev = False
                    
                    # do not use l because it looks like a 1
                    for m in k.drops:
                        
                        device = '[%s %s] %s' % (lot, wafer, 
                                                 k.summary.part_id[0])
                        
                        if fail_num == k.summary.fail_uni[0]:
                            continue
                        
                        if m.summary.failure[0] and not fail_prev:
                            rbm_fail = rbm_fail.append(m.summary)
                            
                            if k.summary.fail_uni[0] == 1:
                                rbm_devs.append(device)
                            else:
                                rbm_devs.append(device + ' (' + str(fail_num + 1) + ')')
                                
                            fail_num = fail_num + 1
                            fail_prev = True
                        else:
                            fail_prev = False
                        
        file_title = 'rbm_plot'
        plot_title = 'RBM Fail Signature'
                        
        rbm_axis = [x+1 for x in range(len(rbm_devs))]
        
        if not rbm_fail.empty:
                        
            plt.figure(figsize=(12,7.5))
            plt.plot(rbm_axis, rbm_fail.rbm_x, marker='o', color='b', 
                     linestyle='None', label='X RBM')
            plt.plot(rbm_axis, rbm_fail.rbm_y, marker='^', color='y', 
                     linestyle='None', label='Y RBM')
            plt.plot(rbm_axis, rbm_fail.rbm_z, marker='s', color='r', 
                     linestyle='None', label='Z RBM')                        
            ax = plt.gca()
            ax.set_title(plot_title)
            ax.set_ylim([-17000, 17000])
            ax.set_ylabel('RBM (cnt)')
            plt.yticks([-16384, -8192, -4096, 0, 4095, 8191, 16383])
            plt.xticks(rbm_axis, rbm_devs, rotation='vertical')
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            
            if save:
                plt.savefig(self.path + '\\' + file_title.upper() + self.file_type, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
    
if __name__ == '__main__':
    tins = [15854, 15856, 15415]
    # tins = [14316]
    test = LIT(tins)
    test.wafer_plot()
    test.sens_plot()
    test.rbm_plot()
    test.board_plot()
    test.summaries()