
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

data_cols = ('partId', 'X_digital', 'Y_digital', 'Z_digital')
sum_cols = ('Part ID', 'Lot ID', 'Wafer', 'Site', 'X Offset (cnt)', 'Y Offset (cnt)', 'Z Offset (cnt)', 'X Offset (mg)', 'Y Offset (mg)', 'Z Offset (mg)')

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

class Sensor(object):

    def __init__(self, data, resolution):
        self.data = pd.DataFrame(columns=data_cols)
        self.summary = pd.DataFrame(columns=sum_cols)
        
        #print(data)
        self.summary.loc[0, 'Part ID']  = int(data.ChipID.unique()[-1])
        self.summary.loc[0, 'Lot ID']  = str(data.LotID.unique()[-1])
        self.summary.loc[0, 'Wafer']  = int(data.WaferNumber.unique()[-1])
        self.summary.loc[0, 'Site']  = int(data.Site.unique()[-1])

        for n, i in enumerate(data.LastKnownOrient.unique()):
            orient_data = data[data.LastKnownOrient == i]
            self.data.loc[n, 'partId']  = int(orient_data.ChipID.unique()[-1])
            self.data.loc[n, 'X_digital']  = int(orient_data.X_digital.unique()[-1])
            self.data.loc[n, 'Y_digital']  = int(orient_data.Y_digital.unique()[-1])
            self.data.loc[n, 'Z_digital']  = int(orient_data.Z_digital.unique()[-1])
            self.data.loc[n, 'Orientation']  = str(orient_data.LastKnownOrient.unique()[-1])


        self.calculate_0g_offset(resolution)

    def calculate_1g_offset(self, resolution):

        for i in ['X', 'Y', 'Z']:

            self.data = self.data[self.data.Orientation == "[+Z]"]

            if i == 'Z':
                expected_offset = 2 ** resolution
            else:
                expected_offset = 0

            offset = ((self.data["%s_digital" % i].item() - expected_offset) / 2**resolution) * 1000

            self.summary.loc[0, "%s Offset (mg)" % i] = offset

    def calculate_0g_offset(self, resolution):

        for i in ['X', 'Y', 'Z']:

            average_offset = self.data[(self.data.Orientation != "[+%s]" % i) &
                                       (self.data.Orientation != "[-%s]" % i)]

            offset = (average_offset["%s_digital" % i].mean() / 2**resolution) * 1000

            self.summary.loc[0, "%s Offset (cnt)" % i] = average_offset["%s_digital" % i].mean()
            self.summary.loc[0, "%s Offset (mg)" % i] = offset

class SMTO(object):

    def __init__(self, df, hour):#, tin):
#        self.data = df
        #df = df[(df.Tin == tin)]
        self.data = df[(df.LastKnownOrient == '[+Z]') |
                       (df.LastKnownOrient == '[-Z]') |
                       (df.LastKnownOrient == '[+Y]') |
                       (df.LastKnownOrient == '[-Y]') |
                       (df.LastKnownOrient == '[+X]') |
                       (df.LastKnownOrient == '[-X]')]
        self.summary = pd.DataFrame(columns=sum_cols)

        self.resolution = int((np.log(self.data[self.data.LastKnownOrient == '[+Z]']['Z_digital'].mean())/np.log(2)) + 0.5)

        #print(self.data)
        self.sensors = []
        for _ in data.ChipID.unique():
            if _ != 0:
                self.sensors.append(Sensor(self.data[self.data.ChipID == _], self.resolution))

        for _ in self.sensors:
            self.summary = self.summary.append(_.summary, ignore_index=True)

        self.summary['Hour'] = hour

        self.char = pd.DataFrame()

        p = ['X Offset (mg)', 'Y Offset (mg)', 'Z Offset (mg)']

        for i in p:

            self.char.at["Average", i] = self.summary[i].mean()
            self.char.at["St. Deviation", i] = self.summary[i].std()
            self.char.at["Maximum", i] = self.summary[i].max()
            self.char.at["Minimum", i] = self.summary[i].min()
            self.char.at["Absolute Avg.", i] = self.summary[i].abs().mean()

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
    data.replace('[0 , -90]', '[+Z]'  , inplace=True)
    data.replace('[0 , -30]', '[+Z/2]', inplace=True)
    data.replace('[0 , 0]'  , '[0Z]'  , inplace=True)
    data.replace('[0 , 30]' , '[-Z/2]', inplace=True)
    data.replace('[0 , 90]' , '[-Z]'  , inplace=True)

    # convert to relative Y
    data.replace('[180 , 0]', '[+Y]'  , inplace=True)
    data.replace('[120 , 0]', '[+Y/2]', inplace=True)
    data.replace('[90 , 0]' , '[0Y]'  , inplace=True)
    data.replace('[60 , 0]' , '[-Y/2]', inplace=True)
    data.replace('[0 , 0]'  , '[-Y]'  , inplace=True)

    # convert to relative X
    data.replace('[90 , 0]' , '[+X]'  , inplace=True)
    data.replace('[30 , 0]' , '[+X/2]', inplace=True)
    data.replace('[0 , 0]'  , '[0X]'  , inplace=True)
    data.replace('[-30 , 0]', '[-X/2]', inplace=True)
    data.replace('[-90 , 0]', '[-X]'  , inplace=True)

    adj_data = pd.DataFrame()

    for sensor in data.ChipID.unique():
        sensor_data = data[data.ChipID == sensor]
        sensor_data.reset_index(drop=True, inplace=True)
        sensor_data.loc[len(sensor_data) - 6, 'LastKnownOrient'] = '[-Y]'
        sensor_data.loc[len(sensor_data) - 5, 'LastKnownOrient'] = '[+X]'
        sensor_data.loc[len(sensor_data) - 3, 'LastKnownOrient'] = '[0X]'
        adj_data = adj_data.append(sensor_data, ignore_index=True)

    return adj_data
    
def average(list): 
    return sum(list) / len(list)

    ##########################################################################


if __name__ == "__main__":

    conn = connect_sql()
    tins = [20268,20270]
    # tins = [20270,20278,20298,20317,20338,20367]
    #tins = [20271,20282,20299,20317,20339,20368]   
    
    for tin in tins:
        print("---------------\n"+str([tin])+"\n---------------")
        data = pull_data(conn, 'lit', '*', [tin])
        adj_data = data
        #adj_data.append(preprocess(data))  # TODO: Confirm with Andrew if this needs to be fixed
        if tin is not tins[0]:
            smto_curr = SMTO(adj_data, 2.5)
            print(smto_curr.char) 
            for i in smto_curr.char.columns.values:
                smto_curr.char.loc['Average', i] = average([smto_curr.char.loc['Average',i] , smto_last.char.loc['Average',i]])
                
                smto_curr.char.loc['St. Deviation',i] = average([smto_curr.char.loc['St. Deviation',i] , smto_last.char.loc['St. Deviation',i]])
                
                smto_curr.char.loc['Maximum',i] = max([smto_curr.char.loc['Maximum',i] , smto_last.char.loc['Maximum',i]])
                
                smto_curr.char.loc['Minimum',i] = min([smto_curr.char.loc['Minimum',i] , smto_last.char.loc['Minimum',i]])
                
                smto_curr.char.loc['Absolute Avg.',i] = average([smto_curr.char.loc['Absolute Avg.',i] , smto_last.char.loc['Absolute Avg.',i]])
        else:
            smto_curr = SMTO(adj_data, 2.5)
            print(smto_curr.char) 
        smto_last = smto_curr
        
    print("---------------\nALL TINS\n---------------")   
    smto = smto_curr
    print(smto.char)
    
    outdir = "C:/Users/"+getpass.getuser()+"/Desktop/PythonAutomatedProcessing/smto/"+str(tins)
    if not os.path.exists(outdir):
        os.makedirs(outdir)    
        
    #adj_data.to_csv(outdir+"/adj_data"+str(tins)+".csv")
        
    smto.char.to_csv(outdir+"/smto"+str(tins)+".csv")

    percentage(smto.summary, "X Offset (mg)", xlim_max=120, xlim_maj_step=40, xlim_min_step=10, ylim=70)
    plt.savefig(outdir+"/x_smto"+str(tins)+".png", bbox_inches='tight')

    percentage(smto.summary, "Y Offset (mg)", xlim_max=120, xlim_maj_step=40, xlim_min_step=10, ylim=70)
    plt.savefig(outdir+"/y_smto"+str(tins)+".png", bbox_inches='tight')

    percentage(smto.summary, "Z Offset (mg)", xlim_max=120, xlim_maj_step=40, xlim_min_step=10, ylim=70)
    plt.savefig(outdir+"/z_smto"+str(tins)+".png", bbox_inches='tight')
