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
import datetime

from tkinter import filedialog
from tkinter import *

date = datetime.datetime.now()
dt = str(date.year)+"_"+str(date.month)+"."+str(date.day)+"_"+str(date.hour)+"."+str(date.minute)


def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    
    return ("%s" % int(100 * y))
    


def dataSummary(data):
    
    char = pd.DataFrame()
    
    p = ["Standby IDD (uA)"]
    
    for i in p:
        char.at["Average", i] = data[i].mean()
        char.at["St. Deviation", i] = data[i].std()
        char.at["Maximum", i] = data[i].max()
        char.at["Minimum", i] = data[i].min()
        #char.at["Abs. Average", i] = data[i].abs().mean()
    print(char)
    
    return char
    

def hist_StandbyIDD(data, outdir):
    fig = plt.figure(figsize=(3.4, 2.25))                                                            
    ax = fig.add_subplot(1,1,1) 
    
    percent_lim = .5 # will be represented as percent (* 100)
    
    xlim_max = 65
    xlim_min = 45
    xlim_maj_step = 5  # must be whole number
    xlim_min_step = .1  # must be whole number
    
    # paramter will be the x labels
    x_maj_ticks = [xlim_min+xlim_maj_step*i for i in itertools.takewhile(lambda x : xlim_min+xlim_maj_step*x <= xlim_max, range(int((2*xlim_max)/xlim_maj_step) + 1))]
    x_min_ticks = [xlim_min+xlim_min_step*i for i in itertools.takewhile(lambda x : xlim_min+xlim_min_step*x <= xlim_max, range(int((2*xlim_max)/xlim_min_step) + 1))]
    
    ylim = 1
    ylim_maj_step = .1  # will be represented as percent (* 100)
    ylim_min_step = .05 # will be represented as percent (* 100)
    
    y_maj_ticks = [ylim_maj_step*i for i in itertools.takewhile(lambda x : ylim_maj_step*x <= ylim, range(int(ylim/ylim_maj_step) + 1))]
    y_min_ticks = [ylim_min_step*i for i in itertools.takewhile(lambda x : ylim_min_step*x <= ylim, range(int(ylim/ylim_min_step) + 1))]
    
    hist, bins = np.histogram(data["Standby IDD (uA)".upper()], bins=x_min_ticks)
    ax.bar(bins[:-1] + 0.5 * (bins[1]-bins[0]), hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]), color='grey', edgecolor='black')

    formatter = FuncFormatter(to_percent)

    # Set the formatter
    plt.gca().yaxis.set_major_formatter(formatter)
    
    plt.xlabel('Standby IDD (uA)'.upper())
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
    
    plt.savefig(outdir+"/Standby Current_"+dt+".png", bbox_inches='tight')
        
        
if __name__ == "__main__":
    
    #import excel file
    root = Tk()
    root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file")
    print('reading excel file ...')
    #look for appropriate headers
    read = pd.read_excel(root.filename, usecols = ['350:idd_standby'])
    #drop null values and non numeric values
    #read = read.dropna()
    read = read[~read.isin(['A'])]
    #rename headers for graphs
    Data= pd.DataFrame()
    Data.columns = ['Sleep IDD (uA)']
    Data = read['350:idd_standby']
    Data.rename(columns = {'350:idd_standby':'Standby IDD (uA)'}, inplace = True)*1000000
    Data = Data.dropna()
    Data.columns = ['Standby IDD (uA)']
    print(Data)
    
    outdir = "C:/Users/"+getpass.getuser()+"/Desktop/PythonAutomatedProcessing/FT/Standby Current"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    #make data summary table
    char = dataSummary(Data)
    char.to_csv(outdir+"/Summary"+dt+".csv")
    print("Data Summary saved to: "+outdir+"/Summary"+dt+".csv")

    #plot the data in a histogram
    hist_StandbyIDD(Data, outdir)
