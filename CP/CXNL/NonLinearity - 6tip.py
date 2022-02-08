import itertools
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import scipy.stats as sps

data_cols = ('partId', 'X_digital', 'Y_digital', 'Z_digital', 'Orientation', 'X_predicted', 'Y_predicted', 'Z_predicted', 'X_lin_error', 'Y_lin_error', 'Z_lin_error')
sum_cols = ('Part ID', 'Lot ID', 'Wafer', 'Site', 'X Non-Linearity (%)', 'Y Non-Linearity (%)', 'Z Non-Linearity (%)')

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
            self.data.loc[n, 'X_digital']  = int(orient_data.X_digital.unique()[-1])
            self.data.loc[n, 'Y_digital']  = int(orient_data.Y_digital.unique()[-1])
            self.data.loc[n, 'Z_digital']  = int(orient_data.Z_digital.unique()[-1])
            self.data.loc[n, 'Orientation']  = str(orient_data.LastKnownOrient.unique()[-1])

        self.calculate_nonlinearity()

    def calculate_nonlinearity(self):
        
        for i in ['X', 'Y', 'Z']:
            
            axis_data = [self.data[self.data.Orientation == "[+%s]" % i]["%s_digital" % i].item(), \
                         self.data[self.data.Orientation == "[+%s/2]" % i]["%s_digital" % i].item(), \
                         self.data[self.data.Orientation == "[0%s]" % i]["%s_digital" % i].item(), \
                         self.data[self.data.Orientation == "[-%s/2]" % i]["%s_digital" % i].item(), \
                         self.data[self.data.Orientation == "[-%s]" % i]["%s_digital" % i].item()]
            
            lr = sps.linregress([1, 0.5, 0, -0.5, -1], axis_data)            
            
            if i == 'Z':
                n = 0
            elif i == 'Y':
                n = 5
            else:
                n = 10
                
            predicted_1g    = lr[1] + lr[0]
            predicted_0p5g  = lr[1] + (lr[0]/2)
            predicted_0g    = lr[1]
            predicted_n0p5g = lr[1] - (lr[0]/2)
            predicted_n1g   = lr[1] - lr[0]
            
            lin_error_1g    = ((axis_data[0] - predicted_1g) / lr[0]) * 1000
            lin_error_0p5g  = ((axis_data[1] - predicted_0p5g) / lr[0]) * 1000
            lin_error_0g    = ((axis_data[2] - predicted_0g) / lr[0]) * 1000
            lin_error_n0p5g = ((axis_data[3] - predicted_n0p5g) / lr[0]) * 1000
            lin_error_n1g   = ((axis_data[4] - predicted_n1g) / lr[0]) * 1000
            
            lin_error =([lin_error_1g, lin_error_0p5g, lin_error_0g, lin_error_n0p5g, lin_error_n1g])
            
            nl = (max(abs(m) for m in lin_error) / 2000) * 100
            
            self.data.set_value(n+0, "%s_predicted" % i, predicted_1g)
            self.data.set_value(n+1, "%s_predicted" % i, predicted_0p5g)
            self.data.set_value(n+2, "%s_predicted" % i, predicted_0g)
            self.data.set_value(n+3, "%s_predicted" % i, predicted_n0p5g)
            self.data.set_value(n+4, "%s_predicted" % i, predicted_n1g)
            
            self.data.set_value(n+0, "%s_lin_error" % i, lin_error_1g)
            self.data.set_value(n+1, "%s_lin_error" % i, lin_error_0p5g)
            self.data.set_value(n+2, "%s_lin_error" % i, lin_error_0g)
            self.data.set_value(n+3, "%s_lin_error" % i, lin_error_n0p5g)
            self.data.set_value(n+4, "%s_lin_error" % i, lin_error_n1g)
            
            self.summary.set_value(0, "%s Non-Linearity (%%)" % i, nl)
            
#            self.data[self.data.Orientation == "[+%s]" % i].loc[0, "%s_predicted" % i]   = lr[1] + lr[0]
#            self.data[self.data.Orientation == "[+%s/2]" % i]["%s_predicted" % i] = lr[1] + (lr[0]/2)
#            self.data[self.data.Orientation == "[0%s]" % i]["%s_predicted" % i]   = lr[1]
#            self.data[self.data.Orientation == "[-%s/2]" % i]["%s_predicted" % i] = lr[1] - (lr[0]/2)
#            self.data[self.data.Orientation == "[-%s]" % i]["%s_predicted" % i]   = lr[1] - lr[0]
                            
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
        
def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    
    return ("%s" % int(100 * y))

class NonLinearity(object):
    
    def __init__(self, data):
        self.data = data
        self.summary = pd.DataFrame(columns=sum_cols)
        
        self.sensors = []
        for _ in data.ChipID.unique():
            if _ != 0:
                self.sensors.append(Sensor(self.data[self.data.ChipID == _]))
            
        for _ in self.sensors:
            self.summary = self.summary.append(_.summary, ignore_index=True)
        
        self.char = pd.DataFrame()
        
        p = ['X Non-Linearity (%)', 'Y Non-Linearity (%)', 'Z Non-Linearity (%)']
        
        for i in p:
            
            self.char.set_value("Average", i, self.summary[i].mean())
            self.char.set_value("St. Deviation", i, self.summary[i].std())
            self.char.set_value("Maximum", i, self.summary[i].max())
            self.char.set_value("Minimum", i, self.summary[i].min())