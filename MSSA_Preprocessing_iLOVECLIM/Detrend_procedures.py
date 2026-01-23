#import warnings
import pylab 
#import math
import numpy as np
import datetime
#import time
#import glob, os

from mssakit.MSSA_config import MSSAConfig
#from mssakit_new.MSSA_config import LevelZonalBottom


def DetrendProcedure_config(config: MSSAConfig, data_all, time):
    """Detrending time series. Order of detrend polynomical is input in MSSAConfig.

    Requires:
    MSSAconfig \n
    data_all \n
    time

    """
    if config.trend_type > 0:
        print('Detrended using a '+str(config.trend_type)+'-order polynomial')
        for series_i in range(len(data_all[0])):
             #Removes the time series for each indiviual time series
             #Take a polynial fit (basic function) to detrend the data
             rank = pylab.polyfit(time, data_all[:, series_i], config.trend_type)	
             fitting = np.zeros(len(time))
             for rank_i in range(len(rank)):
                #Take the fit through the data
                fitting += rank[rank_i] * (time**(len(rank) - 1 - rank_i))
             data_all[:, series_i] = data_all[:, series_i] - fitting
    else:
        print("No detrending applied")


    return data_all
    # season_remove option removed for readability, not used in centennial variability analysis.
    
    '''if config.season_remove == 0: 
        return data_all
    
    #___________________________________
    else:
        print('Seasonal signal is removed\n')

        time_month = np.zeros(len(time))
        for time_i in range(len(time)):
            #Retain the month for each time step
            time_month[time_i] = int(str(datetime.date.fromordinal(int(time[time_i])))[5:7])

        for month_i in range(1, 13):              #Looping over each month

            #Saves all the index with the same month
            index_month = np.where(month_i == time_month)[0] 

            if len(index_month) == 0:
                #No relevant months included
                continue

            #Take for each month the monthly mean over the analysed period
            monthly_mean		= np.mean(data_all[index_month], axis = 0)

            #Subtract monthly mean
            data_all[index_month]	= data_all[index_month] - monthly_mean
    return data_all'''
