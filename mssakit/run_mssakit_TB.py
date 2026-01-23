from mssakit.MSSA_config import MSSAConfig
from Packages.MSSA_Preprocessing_iLOVECLIM.Read_in_Data_iLC import ReadinData_ilc_config
from mssakit.mssakit_TB import mssa_new

import numpy as np

def run_mssa_iloveclim(config: MSSAConfig):

    variables_list = [config.variable,config.variable2, config.variable3, config.variable4, config.variable5 ]

    list_data_all = []
    list_data_std = []
    for i in np.arange(0,config.numberOfVar,1):
        time, dims, data_all,data_std, masked_field = ReadinData_ilc_config(config=config, current_variable=variables_list[i])
        list_data_all.append(data_all)
        list_data_std.append(data_std)
    
    mssa_run = mssa_new(config=config, ts = list_data_all, dims=dims, masked_field=masked_field, t=time) # can move most of these into the config file.. I think
    mssa_run_MC = mssa_run.mcmssa(config=config) 
    mssa_run.storing_spectrum_data(config=config)
    mssa_run.storing_EEOF_data(config=config, list_data_std=list_data_std)
    







def run_MC_MSSA_Proxy_Grid(config: MSSAConfig,list_data_all, list_data_std,  masked_field, time, dims ):
    mssa_run = mssa_new(config=config, ts = list_data_all, dims=dims, masked_field=masked_field, t=time) # can move most of these into the config file.. I think
    mssa_run_MC = mssa_run.mcmssa(config=config) 
    mssa_run.storing_spectrum_data(config=config)
    mssa_run.storing_EEOF_data(config=config, list_data_std=list_data_std)


    
    