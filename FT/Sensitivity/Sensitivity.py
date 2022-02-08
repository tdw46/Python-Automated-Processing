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
    
    p = ["X Sensitivity (LSB/g)", "Y Sensitivity (LSB/g)", "Z Sensitivity (LSB/g)"]
    
    for i in p:
        char.at["Average", i] = data[i].mean()
        char.at["St. Deviation", i] = data[i].std()
        char.at["Maximum", i] = data[i].max()
        char.at["Minimum", i] = data[i].min()
        #char.at["Abs. Average", i] = data[i].abs().mean()
    print(char)
    
    return char
    

def hist_sensitivity(data, outdir):
    for axis in ['X', 'Y', 'Z']:
        fig = plt.figure(figsize=(3.4, 2.25))                                                            
        ax = fig.add_subplot(1,1,1) 
        
        percent_lim = .5 # will be represented as percent (* 100)
        
        xlim_max = 264
        xlim_min = 248
        xlim_maj_step = 8  # must be whole number
        xlim_min_step = 1   # must be whole number
        
        # paramter will be the x labels
        x_maj_ticks = [xlim_min+xlim_maj_step*i for i in itertools.takewhile(lambda x : xlim_min+xlim_maj_step*x <= xlim_max, range(int((2*xlim_max)/xlim_maj_step) + 1))]
        x_min_ticks = [xlim_min+xlim_min_step*i for i in itertools.takewhile(lambda x : xlim_min+xlim_min_step*x <= xlim_max, range(int((2*xlim_max)/xlim_min_step) + 1))]
        
        ylim = 1
        ylim_maj_step = .1  # will be represented as percent (* 100)
        ylim_min_step = .05 # will be represented as percent (* 100)
        
        y_maj_ticks = [ylim_maj_step*i for i in itertools.takewhile(lambda x : ylim_maj_step*x <= ylim, range(int(ylim/ylim_maj_step) + 1))]
        y_min_ticks = [ylim_min_step*i for i in itertools.takewhile(lambda x : ylim_min_step*x <= ylim, range(int(ylim/ylim_min_step) + 1))]
        
        hist, bins = np.histogram(data["%s Sensitivity (LSB/g)" % axis.upper()], bins=x_min_ticks)
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
        
        plt.savefig(outdir+"/"+axis+"sensitivity_"+dt+".png", bbox_inches='tight')
        
        
if __name__ == "__main__":
    
    #import excel file
    root = Tk()
    root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file")
    print('reading excel file ...')
    #look for appropriate headers
    read = pd.read_excel(root.filename, usecols = ['BIN_No','7312:X Sensitivity', '7313:Y Sensitivity', '7314:Z Sensitivity'])
    #drop anything that isn't in bin 1, and then remove the bin column
    Data = read[read.BIN_No == 1].drop(['BIN_No'], axis=1)
    #drop NANs and non numeric results
    Data = Data.dropna()
    Data = Data[~Data.isin(['counts/g'])]
    #rename for plotting
    Data.rename(columns = {'7312:X Sensitivity':'X Sensitivity (LSB/g)', '7313:Y Sensitivity':'Y Sensitivity (LSB/g)', '7314:Z Sensitivity':'Z Sensitivity (LSB/g)'}, inplace = True)
    print(Data)
    
    outdir = "C:/Users/"+getpass.getuser()+"/Desktop/PythonAutomatedProcessing/FT/Sensitivity"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    #make data summary table
    char = dataSummary(Data)
    char.to_csv(outdir+"/Summary"+dt+".csv")
    print("Data Summary saved to: "+outdir+"/Summary"+dt+".csv")

    #plot the data in a histogram
    hist_sensitivity(Data, outdir)
