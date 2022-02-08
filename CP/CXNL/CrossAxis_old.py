
import itertools
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import pymysql as pms

data_cols = ('partId', 'X_RBM', 'Y_RBM', 'Z_RBM', 'X_Digital', 'Y_Digital', 'Z_Digital', 'Orientation')
sum_cols = ('Part ID', 'Lot ID', 'Wafer', 'Site', 
            'Sxy (%)', 'Sxz (%)', 'Sx (%)', \
            'Syx (%)', 'Syz (%)', 'Sy (%)', \
            'Szx (%)', 'Szy (%)', 'Sz (%)')
class Sensor(object):
    
    def __init__(self, df):
        self.data = pd.DataFrame(columns=data_cols)
        self.summary = pd.DataFrame(columns=sum_cols)
        
        self.summary.loc[0, 'Part ID']  = int(df.ChipID.unique()[-1])
        self.summary.loc[0, 'Lot ID']  = str(df.LotID.unique()[-1])
        self.summary.loc[0, 'Wafer']  = int(df.WaferNumber.unique()[-1])
        self.summary.loc[0, 'Site']  = int(df.Site.unique()[-1])
        
        for n, i in enumerate(df.LastKnownOrient.unique()):
            orient_data = df[df.LastKnownOrient == i]
            self.data.loc[n, 'partId']  = int(orient_data.ChipID.unique()[-1])
            self.data.loc[n, 'X_RBM']  = int(orient_data.X_RBM.unique()[-1])
            self.data.loc[n, 'Y_RBM']  = int(orient_data.Y_RBM.unique()[-1])
            self.data.loc[n, 'Z_RBM']  = int(orient_data.Z_RBM.unique()[-1])
            self.data.loc[n, 'X_Digital']  = int(orient_data.X_digital.unique()[-1])
            self.data.loc[n, 'Y_Digital']  = int(orient_data.Y_digital.unique()[-1])
            self.data.loc[n, 'Z_Digital']  = int(orient_data.Z_digital.unique()[-1])
            self.data.loc[n, 'Orientation']  = str(orient_data.LastKnownOrient.unique()[-1])

        self.calculate_cross_axis()

    def calculate_cross_axis(self):
        
        for i in ['X', 'Y', 'Z']:
            
            print(self.data)
            x_rbm_sens = (self.data[self.data.Orientation == "[+%s]" % i]["X_RBM"].item() - self.data[self.data.Orientation == "[-%s]" % i]["X_RBM"].item()) / 2
            y_rbm_sens = (self.data[self.data.Orientation == "[+%s]" % i]["Y_RBM"].item() - self.data[self.data.Orientation == "[-%s]" % i]["Y_RBM"].item()) / 2
            z_rbm_sens = (self.data[self.data.Orientation == "[+%s]" % i]["Z_RBM"].item() - self.data[self.data.Orientation == "[-%s]" % i]["Z_RBM"].item()) / 2

            
#            x_rbm_sens = (self.data[self.data.Orientation == "[+%s]" % i]["X_Digital"].item() - self.data[self.data.Orientation == "[-%s]" % i]["X_Digital"].item()) / 2
#            y_rbm_sens = (self.data[self.data.Orientation == "[+%s]" % i]["Y_Digital"].item() - self.data[self.data.Orientation == "[-%s]" % i]["Y_Digital"].item()) / 2
#            z_rbm_sens = (self.data[self.data.Orientation == "[+%s]" % i]["Z_Digital"].item() - self.data[self.data.Orientation == "[-%s]" % i]["Z_Digital"].item()) / 2

            if i == 'X':
                s_ia = (y_rbm_sens / x_rbm_sens) * 100
                s_ib = (z_rbm_sens / x_rbm_sens) * 100
                s_i =  (((y_rbm_sens ** 2.0) + (z_rbm_sens ** 2.0)) ** (0.5) / x_rbm_sens) * 100
            
                self.summary.set_value(0, "Sxy (%)", s_ia)
                self.summary.set_value(0, "Sxz (%)", s_ib)
                self.summary.set_value(0, "Sx (%)", s_i)
                      
            elif i == 'Y':
                s_ia = (x_rbm_sens / y_rbm_sens) * 100
                s_ib = (z_rbm_sens / y_rbm_sens) * 100
                s_i =  (((x_rbm_sens ** 2) + (z_rbm_sens ** 2)) ** (0.5) / y_rbm_sens) * 100
                      
                self.summary.set_value(0, "Syx (%)", s_ia)
                self.summary.set_value(0, "Syz (%)", s_ib)
                self.summary.set_value(0, "Sy (%)", s_i) 
                     
            else:
                s_ia = (x_rbm_sens / z_rbm_sens) * 100
                s_ib = (y_rbm_sens / z_rbm_sens) * 100
                s_i =  (((x_rbm_sens ** 2) + (y_rbm_sens ** 2)) ** (0.5) / z_rbm_sens) * 100

                self.summary.set_value(0, "Szx (%)", s_ia)
                self.summary.set_value(0, "Szy (%)", s_ib)
                self.summary.set_value(0, "Sz (%)", s_i)
                
def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    
    return ("%s" % int(100 * y))

class CrossAxis(object):
    
    def __init__(self, data):
        
        # remove non 6Tip data
        self.data = data[(data.LastKnownOrient == '[+Z]') |
                         (data.LastKnownOrient == '[-Z]') |
                         (data.LastKnownOrient == '[+Y]') |
                         (data.LastKnownOrient == '[-Y]') |
                         (data.LastKnownOrient == '[+X]') |
                         (data.LastKnownOrient == '[-X]')]

        self.summary = pd.DataFrame(columns=sum_cols)
        
        self.sensors = []
        for _ in data.ChipID.unique():
            if _ != 0:
                self.sensors.append(Sensor(self.data[self.data.ChipID == _]))
                
        for _ in self.sensors:
            self.summary = self.summary.append(_.summary, ignore_index=True)
        
        self.char = pd.DataFrame()
        
        p = ["Sxy (%)", "Sxz (%)", "Syx (%)", "Syz (%)", "Szx (%)", "Szy (%)"]
        
        for i in p:
            
            self.char.set_value("Average", i, self.summary[i].mean())
            self.char.set_value("St. Deviation", i, self.summary[i].std())
            self.char.set_value("Maximum", i, self.summary[i].max())
            self.char.set_value("Minimum", i, self.summary[i].min())
        