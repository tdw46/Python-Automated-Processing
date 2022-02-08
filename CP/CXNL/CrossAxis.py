
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
            x_rbm_sens = (self.data[self.data.Orientation == "[+X]"]["%s_RBM" % i].item() - self.data[self.data.Orientation == "[-X]"]["%s_RBM" % i].item()) / 2  # Sx, Syx, Szx
            y_rbm_sens = (self.data[self.data.Orientation == "[+Y]"]["%s_RBM" % i].item() - self.data[self.data.Orientation == "[-Y]"]["%s_RBM" % i].item()) / 2  # Sxy, Sy, Szy
            z_rbm_sens = (self.data[self.data.Orientation == "[+Z]"]["%s_RBM" % i].item() - self.data[self.data.Orientation == "[-Z]"]["%s_RBM" % i].item()) / 2  # Sxz, Syz, Sz

            
#            x_rbm_sens = (self.data[self.data.Orientation == "[+%s]" % i]["X_Digital"].item() - self.data[self.data.Orientation == "[-%s]" % i]["X_Digital"].item()) / 2
#            y_rbm_sens = (self.data[self.data.Orientation == "[+%s]" % i]["Y_Digital"].item() - self.data[self.data.Orientation == "[-%s]" % i]["Y_Digital"].item()) / 2
#            z_rbm_sens = (self.data[self.data.Orientation == "[+%s]" % i]["Z_Digital"].item() - self.data[self.data.Orientation == "[-%s]" % i]["Z_Digital"].item()) / 2

            if i == 'X':
                s_ia = (y_rbm_sens / x_rbm_sens) * 100  # ( sxy / sx ) * 100
                s_ib = (z_rbm_sens / x_rbm_sens) * 100  # ( Sxz / sx ) * 100
                s_i =  (((y_rbm_sens ** 2.0) + (z_rbm_sens ** 2.0)) ** (0.5) / x_rbm_sens) * 100  # (sqrt(Sxy^2 + Sxz^2)/Sx)*100
            
                self.summary.at[0, "Sxy (%)"] = s_ia
                self.summary.at[0, "Sxz (%)"] = s_ib
                self.summary.at[0, "Sx (%)"] = s_i
                      
            elif i == 'Y':
                s_ia = (x_rbm_sens / y_rbm_sens) * 100  # ( syx / sy )*100
                s_ib = (z_rbm_sens / y_rbm_sens) * 100  # ( syz / sy )*100
                s_i =  (((x_rbm_sens ** 2) + (z_rbm_sens ** 2)) ** (0.5) / y_rbm_sens) * 100  # (sqrt(Syx^2 + Syz^2)/Sy)*100
                      
                self.summary.at[0, "Syx (%)"] = s_ia
                self.summary.at[0, "Syz (%)"] = s_ib
                self.summary.at[0, "Sy (%)"] = s_i
                     
            else:
                s_ia = (x_rbm_sens / z_rbm_sens) * 100 # ( szx / sz )*100
                s_ib = (y_rbm_sens / z_rbm_sens) * 100 # ( szy / sz )*100
                s_i =  (((x_rbm_sens ** 2) + (y_rbm_sens ** 2)) ** (0.5) / z_rbm_sens) * 100  # (sqrt(Szx^2 + Szy^2)/Sz)*100

                self.summary.at[0, "Szx (%)"] = s_ia
                self.summary.at[0, "Szy (%)"] = s_ib
                self.summary.at[0, "Sz (%)"] = s_i
                
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
            
            self.char.at["Average", i] = self.summary[i].mean()
            self.char.at["St. Deviation", i] = self.summary[i].std()
            self.char.at["Maximum", i] = self.summary[i].max()
            self.char.at["Minimum", i] = self.summary[i].min()
        