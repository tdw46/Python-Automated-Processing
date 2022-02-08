# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 21:58:06 2017

@author: "Ronnie"

Modified by Tyler Walker on Fri Mar 6 14:17:09 2020
"""

import configparser as cp
from datetime import datetime
import itertools
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import os
import getpass
import os.path
import pymysql as pms
import pandas as pd

def to_percent(y, position):
    return ("%s" % int(100*y))

def percentage(df, column, xlim_max=100, xlim_min=None, xlim_maj_step=25, xlim_min_step=5, ylim=100, ylim_maj_step=10, ylim_min_step=5):

    fig = plt.figure(figsize=(3.4, 2.25))
    ax = fig.add_subplot(1,1,1)

    if xlim_min is None:
        xlim_min = -xlim_max

    percent_lim = ylim/100

    ylim = 1
    ylim_maj_step /= 100
    ylim_min_step /= 100

    # paramter will be the x labels
    x_maj_ticks = [xlim_min+xlim_maj_step*i for i in itertools.takewhile(lambda x : xlim_min+xlim_maj_step*x <= xlim_max, range(int((2*xlim_max)/xlim_maj_step) + 1))]

    # parameter will also be used for bin size
    x_min_ticks = [xlim_min+xlim_min_step*i for i in itertools.takewhile(lambda x : xlim_min+xlim_min_step*x <= xlim_max, range(int((2*xlim_max)/xlim_min_step) + 1))]

    y_maj_ticks = [ylim_maj_step*i for i in itertools.takewhile(lambda x : ylim_maj_step*x <= ylim, range(int(ylim/ylim_maj_step) + 1))]
    y_min_ticks = [ylim_min_step*i for i in itertools.takewhile(lambda x : ylim_min_step*x <= ylim, range(int(ylim/ylim_min_step) + 1))]

    hist, bins = np.histogram(df[column], bins=x_min_ticks)
    ax.bar(bins[:-1] + 0.5 * (bins[1]-bins[0]), hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]), color='grey', edgecolor='black')

    formatter = FuncFormatter(to_percent)

    # Set the formatter
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.xlabel(column)
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

    return fig


def rand_jitter(arr):
        stdev = max(arr)/15.
        return arr + np.random.randn(len(arr)) * stdev

class Athena(object):
    """
    """

#    path = os.getcwd() + '\\lit_automation_%s\\' % datetime.now().strftime("%y%m%d%H%M")
#    board_path = path + 'boards' + '\\'
#    lot_paths = dict()
#
#
#    file_type = '.png'

    def __init__(self, tins):

#        if not os.path.exists(self.path):
#            os.makedirs(self.path)

        # setup style for matplotlib
#        plt.style.use('ggplot')

        if tins is None:
            raise ValueError('No TINs passed in')

        self.tins = tins

        self.conn = self._connect_sql()

    def _connect_sql(self):
        # TODO(Ronnie): Handle with configparser
        # connect to mysql and pull data
        
        # SJ - '10.10.40.9'
        # TW - '10.30.10.2'

        return pms.connect(host='10.10.40.9', port=3306, user='charuser',
                           passwd='charuser', db='mcubeecodb')

    def _pull_data(self, table, columns, tins):
        # collect info for query
        tins = " or ".join(["tin=%d" % i for i in tins])
        columns = ", ".join(columns)
        query = "SELECT %s FROM %s WHERE %s" % (columns, table, tins)
        return pd.read_sql_query(query, self.conn)

    def _get_wafer(self, partId):
        return (partId >> 24) & 0xFF

    def _twos_complement(self, value, bits):
        if (value & (1 << (bits - 1))) != 0:
            value = value - (1 << bits)
        return value

    def _boxplot(self, df, column, by=None, target=None):

        ax = df.boxplot(column=column, by=by, grid=False)

        plt.suptitle("")
        if target is not None:
            plt.axhline(y=target, c='r', label='Target')

        if by is not None:
            ax.set_title("%s vs %s\nWafer(s) %s" % (column, by, (", ".join([str(i) for i in df["Wafer"].unique()]))))
            ax.set_xlabel(by)
        else:
            ax.set_title("%s\nWafer(s) %s" % (column, (", ".join([str(i) for i in df["Wafer"].unique()]))))

        ax.set_ylabel(column)
        plt.legend()

        return plt.gcf()

    def _hist(self, df, column, bins=10):
        ''' '''

        figure = plt.figure()
        ax = df[column].plot.hist(bins=bins, grid=False, edgecolor='black')

        ax.set_title("%s Histogram\nWafer(s) %s" % (column, (", ".join([str(i) for i in df["Wafer"].unique()]))))
        figure.suptitle("")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")

        return figure

    def char(self, data, columns):

        summary = pd.DataFrame()

        for i in columns:
            summary.at["Average", i] = data[i].mean()
            summary.at["St. Deviation", i] = data[i].std()
            summary.at["Maximum", i] = data[i].max()
            summary.at["Minimum", i] = data[i].min()

        return summary

    def bandgap(self):
        ''' '''
        table = 'bandgap_trim'
        columns = ['partId', 'trim', 'bandgap']

        # pull data from database
        data = self._pull_data(table, columns, self.tins)

        # add wafer to each row in dataframe
        data['Wafer'] = data['partId'].apply(lambda x: self._get_wafer(x))

        # rename columns for easier plotting/labelling
        data.rename(columns={'trim':'Bandgap Trim',
                             'partId' : 'Part ID',
                             'bandgap' : 'Bandgap Voltage (V)'}, inplace=True)

        summary = self.char(data, ['Bandgap Voltage (V)'])

        return data, summary

#
#        data.to_csv("C:\\Users\\"+getpass.getuser()+"\\Desktop\\bandgap.csv")
#
#        bins = np.arange(9) - 0.5
#
#        self._boxplot(data, 'Bandgap Voltage (V)', 'Bandgap Trim', target=1.7)
#        self._hist(data, 'Bandgap Trim', bins)
#
#        if len(data["Wafer"].unique()) > 1:
#            for label, wafer_data in data.groupby('Wafer'):
#
#                self._boxplot(wafer_data, 'Bandgap Voltage (V)', 'Bandgap Trim', target=1.7)
#                self._hist(wafer_data, 'Bandgap Trim', bins)

    def ibias(self):
        ''' '''
        table = 'ibias_trim'
        columns = ['partId', 'trim', 'ibias']

        # pull data from database
        data = self._pull_data(table, columns, self.tins)

        # add wafer to each row in dataframe
        data['Wafer'] = data['partId'].apply(lambda x: self._get_wafer(x))

        # convert ibias two's complement
        data['trim'] = data['trim'].apply(lambda x: self._twos_complement(x, 8))

        # rename columns for easier plotting/labelling
        data.rename(columns={'trim':'IBIAS Trim',
                             'partId' : 'Part ID',
                             'ibias' : 'IBIAS Current (uA)'}, inplace=True)

        summary = self.char(data, ['IBIAS Current (uA)'])

        return data, summary

#        data.to_csv("C:\\Users\\"+getpass.getuser()+"\\Desktop\\ibias.csv")
#
#
#        bins = np.arange(-8,7) - 0.5
#
#        self._boxplot(data, 'IBIAS Current (uA)', 'IBIAS Trim', target=1)
#        self._hist(data, 'IBIAS Trim', bins)
#
#        if len(data["Wafer"].unique()) > 1:
#            for label, wafer_data in data.groupby('Wafer'):
#
#                self._boxplot(wafer_data, 'IBIAS Current (uA)', 'IBIAS Trim', target=1)
#                self._hist(wafer_data, 'IBIAS Trim', bins)

    def clock(self):
        table = 'clk_trim'
        columns = ['partId', 'trim', 'freq']

        # pull data from database
        data = self._pull_data(table, columns, self.tins)

        # add wafer to each row in dataframe
        data['Wafer'] = data['partId'].apply(lambda x: self._get_wafer(x))

        # rename columns for easier plotting/labelling
        data.rename(columns={'trim':'Clock Trim',
                             'partId' : 'Part ID',
                             'freq' : 'Frequency (kHz)'}, inplace=True)

        data["Frequency (kHz)"] /= 1e3

        summary = self.char(data, ['Frequency (kHz)'])

        return data, summary

#        data.to_csv("C:\\Users\\"+getpass.getuser()+"\\Desktop\\clock.csv")
#
#        bins = np.arange(-8,8) - 0.5
#
#        self._boxplot(data, 'Frequency (Hz)', 'Clock Trim', target=640000)
#        self._hist(data, 'Clock Trim', bins)
#
#        if len(data["Wafer"].unique()) > 1:
#            for label, wafer_data in data.groupby('Wafer'):
#
#                self._boxplot(wafer_data, 'Frequency (Hz)', 'Clock Trim', target=640000)
#                self._hist(wafer_data, 'Clock Trim', bins)

    def sdm_cm(self):
        ''' '''
        table = 'sdm_cm_trim'
        columns = ['partId', 'cm_off_x', 'cm_off_y', 'cm_off_z', 'x', 'y', 'z']

        # pull data from database
        data = self._pull_data(table, columns, self.tins)

        # add wafer to each row in dataframe
        data['Wafer'] = data['partId'].apply(lambda x: self._get_wafer(x))

        # rename columns for easier plotting/labelling
        data.rename(columns={'cm_off_x':'X CM Trim',
                             'cm_off_y':'Y CM Trim',
                             'cm_off_z':'Z CM Trim',
                             'x':'X (RBM)',
                             'y':'Y (RBM)',
                             'z':'Z (RBM)',
                             'partId' : 'Part ID'}, inplace=True)

        summary = self.char(data, ['X CM Trim', 'Y CM Trim', 'Z CM Trim'])

        return data, summary

#        data.to_csv("C:\\Users\\"+getpass.getuser()+"\\Desktop\\sdm_cm.csv")
#
#        plt.figure()
#        self._boxplot(data, 'X CM Trim', target=29)
#
#        plt.figure()
#        self._boxplot(data, 'Y CM Trim', target=27)
#
#        plt.figure()
#        self._boxplot(data, 'Z CM Trim', target=83)
#
#        self._hist(data, 'X CM Trim')
#        self._hist(data, 'Y CM Trim')
#        self._hist(data, 'Z CM Trim')
#
#        if len(data["Wafer"].unique()) > 1:
#            for label, wafer_data in data.groupby('Wafer'):
#
#                plt.figure()
#                self._boxplot(wafer_data, 'X CM Trim', target=29)
#
#                plt.figure()
#                self._boxplot(wafer_data, 'Y CM Trim', target=27)
#
#                plt.figure()
#                self._boxplot(wafer_data, 'Z CM Trim', target=83)
#
#                self._hist(wafer_data, 'X CM Trim')
#                self._hist(wafer_data, 'Y CM Trim')
#                self._hist(wafer_data, 'Z CM Trim')


    def sdm_aofs(self):
        ''' '''
        table = 'sdm_aofs_trim'
        columns = ['partId', 'xp', 'xn', 'yp', 'yn', 'zp', 'zn', 'finalX', 'finalY', 'finalZ']

        # pull data from database
        data = self._pull_data(table, columns, self.tins)

        # add wafer to each row in dataframe
        data['Wafer'] = data['partId'].apply(lambda x: self._get_wafer(x))

        # rename columns for easier plotting/labelling
        data.rename(columns={'xp':'XP Trim',
                             'xn':'XN Trim',
                             'yp':'YP Trim',
                             'yn':'YN Trim',
                             'zp':'ZP Trim',
                             'zn':'ZN Trim',
                             'finalX':'X (RBM)',
                             'finalY':'Y (RBM)',
                             'finalZ':'Z (RBM)',
                             'partId' : 'Part ID'}, inplace=True)

        summary = self.char(data, ['XP Trim', 'XN Trim', 'YP Trim', 'YN Trim', 'ZP Trim', 'ZN Trim'])

        return data, summary

#        data.to_csv("C:\\Users\\"+getpass.getuser()+"\\Desktop\\sdm_aofs.csv")
#
#        plt.figure()
#        self._boxplot(data, ['XP Trim','XN Trim'], target=29)
#
#        plt.figure()
#        self._boxplot(data, ['YP Trim','YN Trim'], target=27)
#
#        plt.figure()
#        self._boxplot(data, ['ZP Trim','ZN Trim'], target=83)
#
#        self._hist(data, 'XP Trim')
#        self._hist(data, 'YP Trim')
#        self._hist(data, 'ZP Trim')
#
#        if len(data["Wafer"].unique()) > 1:
#            for label, wafer_data in data.groupby('Wafer'):
#
#                plt.figure()
#                self._boxplot(wafer_data, ['XP Trim','XN Trim'], target=29)
#
#                plt.figure()
#                self._boxplot(wafer_data, ['YP Trim','YN Trim'], target=27)
#
#                plt.figure()
#                self._boxplot(wafer_data, ['ZP Trim','ZN Trim'], target=83)
#
#                self._hist(wafer_data, 'XP Trim')
#                self._hist(wafer_data, 'YP Trim')
#                self._hist(wafer_data, 'ZP Trim')

    def mems_sens(self):
        ''' '''
        table = 'mems_sens'
        columns = ['partId', 'Xoff', 'Yoff', 'Zoff', 'Xsens', 'Ysens', 'Zsens']

        # pull data from database
        data = self._pull_data(table, columns, self.tins)

        # add wafer to each row in dataframe
        data['Wafer'] = data['partId'].apply(lambda x: self._get_wafer(x))

        # rename columns for easier plotting/labelling
        data.rename(columns={'Xoff':'X Offset',
                             'Yoff':'Y Offset',
                             'Zoff':'Z Offset',
                             'Xsens':'X Sensitivity (fF/g)',
                             'Ysens':'Y Sensitivity (fF/g)',
                             'Zsens':'Z Sensitivity (fF/g)',
                             'partId' : 'Part ID'}, inplace=True)

        summary = self.char(data, ['X Offset', 'Y Offset', 'Z Offset', 'X Sensitivity (fF/g)', 'Y Sensitivity (fF/g)', 'Z Sensitivity (fF/g)'])

        return data, summary
#
#        data.to_csv("C:\\Users\\"+getpass.getuser()+"\\Desktop\\mems_sens.csv")
#
#        plt.figure()
#        self._boxplot(data, ['X Offset'], target=0)
#
#        plt.figure()
#        self._boxplot(data, ['Y Offset'], target=0)
#
#        plt.figure()
#        self._boxplot(data, ['Z Offset'], target=0)
#
#        plt.figure()
#        self._boxplot(data, ['X Sensitivity (fF/g)'], target=1.4)
#
#        plt.figure()
#        self._boxplot(data, ['Y Sensitivity (fF/g)'], target=1.4)
#
#        plt.figure()
#        self._boxplot(data, ['Z Sensitivity (fF/g)'], target=1.4)
#
#        self._hist(data, 'X Offset')
#        self._hist(data, 'Y Offset')
#        self._hist(data, 'Z Offset')
#        self._hist(data, 'X Sensitivity (fF/g)')
#        self._hist(data, 'Y Sensitivity (fF/g)')
#        self._hist(data, 'Z Sensitivity (fF/g)')
#
#        if len(data["Wafer"].unique()) > 1:
#            for label, wafer_data in data.groupby('Wafer'):
#
#                plt.figure()
#                self._boxplot(wafer_data, ['X Offset'], target=0)
#
#                plt.figure()
#                self._boxplot(wafer_data, ['Y Offset'], target=0)
#
#                plt.figure()
#                self._boxplot(wafer_data, ['Z Offset'], target=0)
#
#                plt.figure()
#                self._boxplot(wafer_data, ['X Sensitivity (fF/g)'], target=1.4)
#
#                plt.figure()
#                self._boxplot(wafer_data, ['Y Sensitivity (fF/g)'], target=1.4)
#
#                plt.figure()
#                self._boxplot(wafer_data, ['Z Sensitivity (fF/g)'], target=1.4)
#
#                self._hist(wafer_data, 'X Offset')
#                self._hist(wafer_data, 'Y Offset')
#                self._hist(wafer_data, 'Z Offset')
#                self._hist(wafer_data, 'X Sensitivity (fF/g)')
#                self._hist(wafer_data, 'Y Sensitivity (fF/g)')
#                self._hist(wafer_data, 'Z Sensitivity (fF/g)')

    def dig_sens(self):
        ''' '''
        table = 'dig_sens'
        columns = ['partId', 'XoffMeas', 'YoffMeas', 'ZoffMeas', 'XsensMeas', 'YsensMeas', 'ZsensMeas']

        # pull data from database
        data = self._pull_data(table, columns, self.tins)

        # add wafer to each row in dataframe
        data['Wafer'] = data['partId'].apply(lambda x: self._get_wafer(x))

        # rename columns for easier plotting/labelling
        data.rename(columns={'XoffMeas':'X Offset',
                             'YoffMeas':'Y Offset',
                             'ZoffMeas':'Z Offset',
                             'XsensMeas':'X Sensitivity',
                             'YsensMeas':'Y Sensitivity',
                             'ZsensMeas':'Z Sensitivity',
                             'partId' : 'Part ID'}, inplace=True)

        summary = self.char(data, ['X Offset', 'Y Offset', 'Z Offset', 'X Sensitivity', 'Y Sensitivity', 'Z Sensitivity'])

        return data, summary

#        data.to_csv("C:\\Users\\"+getpass.getuser()+"\\Desktop\\dig_sens.csv")
#
#        plt.figure()
#        self._boxplot(data, ['X Offset'], target=0)
#
#        plt.figure()
#        self._boxplot(data, ['Y Offset'], target=0)
#
#        plt.figure()
#        self._boxplot(data, ['Z Offset'], target=0)
#
#        plt.figure()
#        self._boxplot(data, ['X Sensitivity'], target=64)
#
#        plt.figure()
#        self._boxplot(data, ['Y Sensitivity'], target=64)
#
#        plt.figure()
#        self._boxplot(data, ['Z Sensitivity'], target=64)
#
#        self._hist(data, 'X Offset')
#        self._hist(data, 'Y Offset')
#        self._hist(data, 'Z Offset')
#        self._hist(data, 'X Sensitivity')
#        self._hist(data, 'Y Sensitivity')
#        self._hist(data, 'Z Sensitivity')
#
#        if len(data["Wafer"].unique()) > 1:
#            for label, wafer_data in data.groupby('Wafer'):
#
#                plt.figure()
#                self._boxplot(wafer_data, ['X Offset'], target=0)
#
#                plt.figure()
#                self._boxplot(wafer_data, ['Y Offset'], target=0)
#
#                plt.figure()
#                self._boxplot(wafer_data, ['Z Offset'], target=0)
#
#                plt.figure()
#                self._boxplot(wafer_data, ['X Sensitivity'], target=64)
#
#                plt.figure()
#                self._boxplot(wafer_data, ['Y Sensitivity'], target=64)
#
#                plt.figure()
#                self._boxplot(wafer_data, ['Z Sensitivity'], target=64)
#
#                self._hist(wafer_data, 'X Offset')
#                self._hist(wafer_data, 'Y Offset')
#                self._hist(wafer_data, 'Z Offset')
#                self._hist(wafer_data, 'X Sensitivity')
#                self._hist(wafer_data, 'Y Sensitivity')
#                self._hist(wafer_data, 'Z Sensitivity')

if __name__ == '__main__':
#    tins = [13268, 13269, 13270, 13271, 13272]
    tins = [14804, 14805] # Mensa
    athena = Athena(tins)
    outdir = "C:/Users/"+getpass.getuser()+"/Desktop/PythonAutomatedProcessing/Athena/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # BANDGAP
    bandgap_data, bandgap_char = athena.bandgap()
    bandgap_char.to_csv(outdir+"bandgap_char.csv")
    percentage(bandgap_data, 'Bandgap Voltage (V)', xlim_max=1.8, xlim_min=1.6, xlim_maj_step=.05, xlim_min_step=.025, ylim=60)
    plt.savefig(outdir+"bandgap.png", bbox_inches='tight')

    # IBIAS
    ibias_data, ibias_char = athena.ibias()
    ibias_char.to_csv(outdir+"ibias_char.csv")
    percentage(ibias_data, 'IBIAS Current (uA)', xlim_max=1.05, xlim_min=0.95, xlim_maj_step=.025, xlim_min_step=.005, ylim=20, ylim_min_step=2)
    plt.savefig(outdir+"ibias.png", bbox_inches='tight')

    # CLOCK
    clock_data, clock_char = athena.clock()
    clock_char.to_csv(outdir+"clock_char.csv")
    percentage(clock_data, 'Frequency (kHz)', xlim_max=650, xlim_min=630, xlim_maj_step=5, xlim_min_step=1, ylim=20, ylim_min_step=2)
    plt.savefig(outdir+"clock.png", bbox_inches='tight')

    # SDM_CM
    sdm_cm_data, sdm_cm_char = athena.sdm_cm()
    sdm_cm_char.to_csv(outdir+"sdm_cm_char.csv")
    percentage(sdm_cm_data, 'X CM Trim', xlim_max=35, xlim_min=15, xlim_maj_step=5, xlim_min_step=1, ylim=50)
    plt.savefig(outdir+"x_sdm_cm.png", bbox_inches='tight')
    percentage(sdm_cm_data, 'Y CM Trim', xlim_max=35, xlim_min=15, xlim_maj_step=5, xlim_min_step=1, ylim=50)
    plt.savefig(outdir+"y_sdm_cm.png", bbox_inches='tight')
    percentage(sdm_cm_data, 'Z CM Trim', xlim_max=90, xlim_min=70, xlim_maj_step=5, xlim_min_step=1, ylim=30, ylim_min_step=2)
    plt.savefig(outdir+"z_sdm_cm.png", bbox_inches='tight')

    # SDM_AOFS
    sdm_aofs_data, sdm_aofs_char = athena.sdm_aofs()
    sdm_aofs_char.to_csv(outdir+"sdm_aofs_char.csv")
    percentage(sdm_aofs_data, 'XP Trim', xlim_max=35, xlim_min=15, xlim_maj_step=5, xlim_min_step=1, ylim=70)
    plt.savefig(outdir+"xp_sdm_aofs.png", bbox_inches='tight')
    percentage(sdm_aofs_data, 'XN Trim', xlim_max=35, xlim_min=15, xlim_maj_step=5, xlim_min_step=1, ylim=70)
    plt.savefig(outdir+"xn_sdm_aofs.png", bbox_inches='tight')
    percentage(sdm_aofs_data, 'YP Trim', xlim_max=35, xlim_min=15, xlim_maj_step=5, xlim_min_step=1, ylim=70)
    plt.savefig(outdir+"yp_sdm_aofs.png", bbox_inches='tight')
    percentage(sdm_aofs_data, 'YN Trim', xlim_max=35, xlim_min=15, xlim_maj_step=5, xlim_min_step=1, ylim=70)
    plt.savefig(outdir+"yn_sdm_aofs.png", bbox_inches='tight')
    percentage(sdm_aofs_data, 'ZP Trim', xlim_max=90, xlim_min=70, xlim_maj_step=5, xlim_min_step=1, ylim=70)
    plt.savefig(outdir+"zp_sdm_aofs.png", bbox_inches='tight')
    percentage(sdm_aofs_data, 'ZN Trim', xlim_max=90, xlim_min=70, xlim_maj_step=5, xlim_min_step=1, ylim=70)
    plt.savefig(outdir+"zn_sdm_aofs.png", bbox_inches='tight')

    # MEMS_SENS
    mems_sens_data, mems_sens_char = athena.mems_sens()
    mems_sens_char.to_csv(outdir+"mems_sens_char.csv")
    percentage(mems_sens_data, 'X Sensitivity (fF/g)', xlim_max=2.0, xlim_min=0.8, xlim_maj_step=0.2, xlim_min_step=0.1, ylim=70)
    plt.savefig(outdir+"x_mems_sens.png", bbox_inches='tight')
    percentage(mems_sens_data, 'Y Sensitivity (fF/g)', xlim_max=2.0, xlim_min=0.8, xlim_maj_step=0.2, xlim_min_step=0.1, ylim=70)
    plt.savefig(outdir+"y_mems_sens.png", bbox_inches='tight')
    percentage(mems_sens_data, 'Z Sensitivity (fF/g)', xlim_max=2.0, xlim_min=0.8, xlim_maj_step=0.2, xlim_min_step=0.1, ylim=70)
    plt.savefig(outdir+"z_mems_sens.png", bbox_inches='tight')

    # DIG_SENS
    dig_sens_data, dig_sens_char = athena.dig_sens()
    dig_sens_char.to_csv(outdir+"dig_sens_char.csv")
#    percentage(dig_sens_data, 'X Offset', xlim_max=1, xlim_min=-1, xlim_maj_step=0.5, xlim_min_step=0.1, ylim=70)
#    plt.savefig(outdir+"bandgap.png", bbox_inches='tight')
#    percentage(dig_sens_data, 'Y Offset', xlim_max=1, xlim_min=-1, xlim_maj_step=0.5, xlim_min_step=0.1, ylim=70)
#    plt.savefig(outdir+"bandgap.png", bbox_inches='tight')
#    percentage(dig_sens_data, 'Z Offset', xlim_max=1, xlim_min=-1, xlim_maj_step=0.5, xlim_min_step=0.1, ylim=70)
#    plt.savefig(outdir+"bandgap.png", bbox_inches='tight')
    percentage(dig_sens_data, 'X Sensitivity', xlim_max=72, xlim_min=56, xlim_maj_step=4, xlim_min_step=1, ylim=60)
    plt.savefig(outdir+"x_dig_sens.png", bbox_inches='tight')
    percentage(dig_sens_data, 'Y Sensitivity', xlim_max=72, xlim_min=56, xlim_maj_step=4, xlim_min_step=1, ylim=60)
    plt.savefig(outdir+"y_dig_sens.png", bbox_inches='tight')
    percentage(dig_sens_data, 'Z Sensitivity', xlim_max=72, xlim_min=56, xlim_maj_step=4, xlim_min_step=1, ylim=60)
    plt.savefig(outdir+"z_dig_sens.png", bbox_inches='tight')
