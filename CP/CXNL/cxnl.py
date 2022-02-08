
import configparser as cp
from datetime import datetime
import itertools
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import os
import pymysql as pms
import pandas as pd
import getpass
import os.path

from CrossAxis import CrossAxis
from NonLinearity import NonLinearity
from ZeroGOffset import ZeroGOffset

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
    
    ##########################################################################
    
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
    data.replace('[180 , 0]', '[-X]'  , inplace=True)
    data.replace('[120 , 0]', '[-X/2]', inplace=True)
    data.replace('[90 , 0]' , '[0X]'  , inplace=True)
    data.replace('[60 , 0]' , '[+X/2]', inplace=True)
    data.replace('[0 , 0]'  , '[+X]'  , inplace=True)
    
    # convert to relative X
    data.replace('[90 , 0]' , '[+Y]'  , inplace=True)
    data.replace('[30 , 0]' , '[+Y/2]', inplace=True)
    data.replace('[0 , 0]'  , '[0Y]'  , inplace=True)
    data.replace('[-30 , 0]', '[-Y/2]', inplace=True)
    data.replace('[-90 , 0]', '[-Y]'  , inplace=True)
    
    adj_data = pd.DataFrame()
    
    for sensor in data.ChipID.unique():
        sensor_data = data[data.ChipID == sensor]
        sensor_data.reset_index(drop=True, inplace=True)
        sensor_data.loc[len(sensor_data) - 6, 'LastKnownOrient'] = '[+X]'
        sensor_data.loc[len(sensor_data) - 5, 'LastKnownOrient'] = '[+Y]'
        sensor_data.loc[len(sensor_data) - 3, 'LastKnownOrient'] = '[0Y]'
        adj_data = adj_data.append(sensor_data, ignore_index=True)
    
    return adj_data    
    
    ##########################################################################
    
if __name__ == '__main__':
    tins = [20415, 20416]

    conn = connect_sql()
    data = pull_data(conn, 'lit', '*', tins)
    adj_data = preprocess(data)
    
    cx = None
    nl = None
    zgo = None

    cx = CrossAxis(adj_data)
    nl = NonLinearity(adj_data)
    # zgo = ZeroGOffset(adj_data)

#      NONLINEARITY

    outdir = "C:/Users/"+getpass.getuser()+"/Desktop/PythonAutomatedProcessing/nonlinearity/"+str(tins)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    adj_data.to_csv(outdir+"/nl"+str(tins)+"adj_data.csv")
    
    if nl is not None:

        nl.char.to_csv(outdir+"/nl"+str(tins)+".csv")
    
        percentage(nl.summary, "X Non-Linearity (%)", xlim_max=2, xlim_min=0, xlim_maj_step=1, xlim_min_step=0.20, ylim=60)
        plt.savefig(outdir+"/nl_x.png", bbox_inches='tight')
    
        percentage(nl.summary, "Y Non-Linearity (%)", xlim_max=2, xlim_min=0, xlim_maj_step=1, xlim_min_step=0.20, ylim=60)
        plt.savefig(outdir+"/nl_y.png", bbox_inches='tight')
    
        percentage(nl.summary, "Z Non-Linearity (%)", xlim_max=2, xlim_min=0, xlim_maj_step=1, xlim_min_step=0.20, ylim=60)
        plt.savefig(outdir+"/nl_z.png", bbox_inches='tight')
        
#     CROSS AXIS
    
    if cx is not None:
    
        cx.char.to_csv(outdir+"/cx"+str(tins)+".csv")
        
        cx_ylim = 80
    
        percentage(cx.summary, "Sxy (%)", xlim_max=6, xlim_maj_step=2, xlim_min_step=0.5, ylim=cx_ylim)
        plt.savefig(outdir+"/cx_sxy.png", bbox_inches='tight')
    
        percentage(cx.summary, "Sxz (%)", xlim_max=6, xlim_maj_step=2, xlim_min_step=0.5, ylim=cx_ylim)
        plt.savefig(outdir+"/cx_sxz.png", bbox_inches='tight')
    
        percentage(cx.summary, "Syx (%)", xlim_max=6, xlim_maj_step=2, xlim_min_step=0.5, ylim=cx_ylim)
        plt.savefig(outdir+"/cx_syx.png", bbox_inches='tight')
    
        percentage(cx.summary, "Syz (%)", xlim_max=6, xlim_maj_step=2, xlim_min_step=0.5, ylim=cx_ylim)
        plt.savefig(outdir+"/cx_syz.png", bbox_inches='tight')
    
        percentage(cx.summary, "Szx (%)", xlim_max=6, xlim_maj_step=2, xlim_min_step=0.5, ylim=cx_ylim)
        plt.savefig(outdir+"/cx_szx.png", bbox_inches='tight')
    
        percentage(cx.summary, "Szy (%)", xlim_max=6, xlim_maj_step=2, xlim_min_step=0.5, ylim=cx_ylim)
        plt.savefig(outdir+"/cx_szy.png", bbox_inches='tight')
    
    # #ZERO G OFFSET , we get this from the SMTO program already, so no need to collect it here.
    
    if zgo is not None:
       
       zgo.char.to_csv(outdir+"/zgo"+str(tins)+".csv")
       
       percentage(zgo.summary, "X Offset (mg)", xlim_max=100, xlim_maj_step=25, xlim_min_step=5, ylim=60)
       plt.savefig(outdir+"/zgo_x.png", bbox_inches='tight')
       
       percentage(zgo.summary, "Y Offset (mg)", xlim_max=100, xlim_maj_step=25, xlim_min_step=5, ylim=60)
       plt.savefig(outdir+"/zgo_y.png", bbox_inches='tight')
       
       percentage(zgo.summary, "Z Offset (mg)", xlim_max=100, xlim_maj_step=25, xlim_min_step=5, ylim=60)
       plt.savefig(outdir+"/zgo_z.png", bbox_inches='tight')
   
       percentage(zgo.summary, "X Sensitivity (LSB/g)", xlim_max=264, xlim_min=248,  xlim_maj_step=8, xlim_min_step=1, ylim=50)
       plt.savefig(outdir+"/sens_x.png", bbox_inches='tight')
   
       percentage(zgo.summary, "Y Sensitivity (LSB/g)", xlim_max=264, xlim_min=248, xlim_maj_step=8, xlim_min_step=1, ylim=50)
       plt.savefig(outdir+"/sens_y.png", bbox_inches='tight')
   
       percentage(zgo.summary, "Z Sensitivity (LSB/g)", xlim_max=264, xlim_min=248, xlim_maj_step=8, xlim_min_step=1, ylim=50)
       plt.savefig(outdir+"/sens_z.png", bbox_inches='tight')
   