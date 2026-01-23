import netCDF4 as netcdf
import numpy as np
import warnings

import os

from mssakit.MSSA_config import MSSAConfig
from mssakit.MSSA_config import LevelZonalBottom


from Packages.MSSA_Preprocessing_iLOVECLIM.Detrend_procedures import DetrendProcedure_config
from Packages.MSSA_Preprocessing_iLOVECLIM.Loading_Proxies_mask import retrieving_proxy_mask


def vertical_averaging_3D_to_2D(data_masked_with_bool, volume, depth_range_index_list): # or 4D to 3D
    ''' data must be an array of shape (time, level, lat/lon, lat/lon)'''
    
    data = data_masked_with_bool
    lower_i = depth_range_index_list[0]
    upper_i = depth_range_index_list[1]


    data_arr_no_time = data[0,:,:,:].filled(np.nan) #mask volume similar to data mask
    ones_arr_no_times = np.where((np.abs(data_arr_no_time) > 0), 1,0) #1 where gridcells have a value that is not 0 or nan.
    vol_masked = ones_arr_no_times * volume   
    
    data_sliced = data[:,lower_i:upper_i, :,:]
    vol_masked_sliced = vol_masked[lower_i:upper_i,:,:]

    Volume_weighted_var = (data_sliced * vol_masked_sliced / vol_masked_sliced.sum(axis=0) ).sum(axis=1)            # (cell_data * cell_vol / depth_integrated_volume_cell ) integrate over depth!
    return Volume_weighted_var





def ReadinData_ilc_config(config: MSSAConfig, current_variable):
    """
    Reads in and pre_processes data from iLOVECLIM output in order to run data with mssakit_TB.

    Input:
    config : MSSAConfig file.
    current_variable : variable to work on. 
    
    output:
     time, dims, data_all, data_std,masked_field	
    """


    print(config.get_filename())
    VAR_data = netcdf.Dataset(config.data_path, 'r')
    time 		= VAR_data.variables['time'][:]
    lat 		= VAR_data.variables['lat'][:]
    lon 		= VAR_data.variables['lon'][:]
    depth 		= VAR_data.variables['level'][:]

    #v_mask = VAR_data.variables['v'][:][0,:,:,:]

    if current_variable == 'temp': 
        var		= VAR_data.variables['temp'][:]
    elif current_variable == 'salt':
        var      =  VAR_data.variables['salt'][:]
    elif current_variable == 'v': 
        var = VAR_data.variables['v'][:]
    elif current_variable == 'u': 
        var = VAR_data.variables['u'][:]        
    elif current_variable == 'rho':
        #rho_filename = '/Users/toonbense/Documents/local_Data_test/' + config.label + '_density.nc'
        #dir_path = os.path.dirname(config.data_path) +"/"
        #rho_filename =  dir_path + config.label + '_density.nc'
        rho_filename = config.rho_filename
        rho_data = netcdf.Dataset(rho_filename, 'r')
        var = rho_data.variables['density25'][:]  ## 'density25' is based on Jackett et al. 2006 / Mehling et al 2023 (EOS with 25 polynomial terms),
        #                                           'TEOS10_density' based on TEOS10 is most accurate, 'TEOS10_sigma0' potential density ref level 0, 'clio_density' based on CLIO documentation (simplest EOS)
        rho_data.close()

        print('currently selected rho = ' + 'density25')
    elif current_variable == 'mld':
        #dir_path = os.path.dirname(config.data_path) +"/"
        #mld_filename = dir_path + config.label + '_mld.nc'
        mld_filename = config.mld_filename
        mld_data = netcdf.Dataset(mld_filename, 'r')
        var1 = mld_data.variables['level'][:] ## level here refers to the depth of the deepest cell within sigma0 0.03 of the second shallowest cell sigma0 ()
        var = np.transpose(var1,(2,0,1))
    elif current_variable == 'd13C': 
        odic_data		= VAR_data.variables['odic'][:]    
        oc13_data		= VAR_data.variables['oc13'][:]
        var = oc13_data/odic_data
    elif current_variable == 'speed': 
        uvel_data		= VAR_data.variables['u'][:]    
        vvel_data		= VAR_data.variables['v'][:]    
        var = np.sqrt(np.square(uvel_data)+np.square(vvel_data))
    elif current_variable == 'PaTh': 
        papart_data = VAR_data.variables['papart'][:]
        thpart_data = VAR_data.variables['thpart'][:]
        var = papart_data/thpart_data
    else:
        print("variable doesn't exist, please choose one of the following: 'temp', 'salt','u','v', 'rho', 'mld','d13C', 'speed','PaTh'")
    VAR_data.close()

    #start_time, end_time, red_time,sel_region,lev_or_zonal, variable,file_name, label = None

        # load in masks
    ds_masks = netcdf.Dataset('/Users/toonbense/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/PhD/iLOVECLIM_code_TB/masks_ilc_tb.nc', 'r')
    vol = ds_masks.variables['Volume'][:]
    speed_mask = ds_masks.variables['speed_3D_mask'][:]

    if config.sel_region == 'Atlantic_Arctic':
        mask_1 = ds_masks.variables['mask_atl_arc'][:]
    elif config.sel_region == 'Atlantic':
        mask_1 = ds_masks.variables['mask_atlantic'][:]
    elif config.sel_region == 'West_Atlantic':
            mask_1 = ds_masks.variables['mask_west_atlantic'][:]
    elif config.sel_region == 'East_Atlantic':
            mask_1 = ds_masks.variables['mask_esat_atlantic'][:]
    elif config.sel_region == 'Arctic':
        mask_1 = ds_masks.variables['mask_arctic'][:] 
    elif config.sel_region == 'Atlantic_Arctic_Southern':
        mask_1 = ds_masks.variables['mask_atl_arc_so'][:]
    elif config.sel_region == "Southern":
        mask_1 = ds_masks.variables['mask_so'][:]
    elif config.sel_region == "Global_Ocean":
        mask_1 = ds_masks.variables['mask_global_ocean'][:]
    elif  config.sel_region == 'SST_proxies':
        mask_1 = retrieving_proxy_mask(proxy_cell_pct=config.proxy_cell_pct, bool180E_180W=False)

    else:
        print("incorrect mask name. Use one of Atlantic, West_Atlantic, East_Atlantic, Arctic, Atlantic_Arctic,Southern, or Atlantic_Arctic_Southern, Global_Ocean", "SST_proxies")
    ds_masks.close()


    
    if (('speed' in config.get_list_variables()) | ('u' in config.get_list_variables())| ('v' in config.get_list_variables())):
        var = var.filled(np.nan)

        var = var * speed_mask
        var[var==0] = np.nan        
        print('Selected only variable values where there are u/v data too.')

    #print(var)


    # Depending on function options, select region, single vertical level or zonal mean
    if config.level_zonal_bottom == LevelZonalBottom.LEVEL.value: # means lat x lon at depth_index config.level_index 
        #var = var.filled(np.nan)
        var_masked = var * mask_1
        if len(var.shape) == 4:
            var_masked = var_masked[:,config.level_index,:,:]
        else:
            print("no Z axis -> continue with var_masked")
        #var_masked[var_masked == 0] = np.nan
        #var_masked[var_masked == -0] = np.nan

        dim_i = lon
        dim_j = lat
    elif config.level_zonal_bottom == LevelZonalBottom.VOL.value: # means lat x lon at depth_index config.level_index 
        #var = var.filled(np.nan)
        var_masked_unweighted = var * mask_1
        if len(var.shape) == 4:
            var_masked = vertical_averaging_3D_to_2D(var_masked_unweighted, volume=vol, depth_range_index_list=config.level_index_range)
            print(f'selecting latxlon values between {depth[config.level_index_range[0]]} and {config.level_index_range[1]} m depth')
        
        elif current_variable == 'mld': # mld has no bottom cells as it is a 2d array. no need to select bottom cells.
            var_masked = var * mask_1
            
        else:
            print("Error, no Z-axis to volume average over...")
            return
        #var_masked[var_masked == 0] = np.nan
        #var_masked[var_masked == -0] = np.nan

        dim_i = lon
        dim_j = lat
        
    elif config.level_zonal_bottom == LevelZonalBottom.ZONAL.value: #lat x depth
        #var = var.filled(np.nan)    
        var_masked = var * mask_1
        var_masked[var_masked==0] = np.nan
        var_masked = np.nanmean(var_masked,axis=3)
        dim_i = lat
        dim_j = depth
        warnings.filterwarnings('ignore')
        
    elif config.level_zonal_bottom == LevelZonalBottom.BOTTOM.value: #Bottom cells
        ds_masks = netcdf.Dataset('/Users/toonbense/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/iLOVECLIM_code_TB/masks_ilc_tb.nc', 'r')
        mask_bottom_cells = ds_masks.variables['mask_deepest_cell_3d'][:]
        ds_masks.close()
        if current_variable == 'mld': # mld has no bottom cells as it is a 2d array. no need to select bottom cells.
            var_masked = var * mask_1
        else:
            var_masked = ((var * mask_bottom_cells).sum(axis=1) ) * mask_1

        dim_i = lon
        dim_j = lat

    elif config.level_zonal_bottom == LevelZonalBottom.Three_D.value: #Bottom cells
        print(LevelZonalBottom)
        #ds_masks = netcdf.Dataset('/Users/toonbense/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/iLOVECLIM_code_TB/masks_ilc_tb.nc', 'r')
        var_masked = var * mask_1
        dim_i = lon
        dim_j = lat
        dim_k = depth

    #print(var_masked.shape)

    # Correct longitudes
    if ((config.level_zonal_bottom==LevelZonalBottom.LEVEL.value) |(config.level_zonal_bottom==LevelZonalBottom.BOTTOM.value) |(config.level_zonal_bottom==LevelZonalBottom.Three_D.value) |(config.level_zonal_bottom==LevelZonalBottom.VOL.value) ):  
        dim_i = np.where(dim_i>360,dim_i-360,dim_i)
        dim_i = np.where(dim_i>180,dim_i-360,dim_i)


    
	#Only retain the selected time restriction
    time	= time[config.start_time: config.end_time]
    var_masked	= var_masked[config.start_time : config.end_time]
    #print(f"var masked shape = {var_masked.shape}")
    print("manually overwriting time parameter from start_time +1 to start_time + end_time +1.")
    time = np.arange(config.start_time+1, config.end_time+1,1)

	#Deduce all the masked elements in the data
    #if len(var_masked.shape == 3):
        #masked_field = var_masked[0,:,:].filled(np.nan)
    masked_field = var_masked[0].filled(np.nan) # select time = 0, and select masked field in 2d or 3d
    masked_field[masked_field == 0 ] = np.nan
    masked_field
    #elif len(var_masked.shape == 4): # or / AND (config.level_zonal_bottom==LevelZonalBottom.Three_D.value)

	#Generate empty array, where only gricells with data are saved in a timexdata array (so no masked elements)
    data_all	= np.zeros((len(time), np.count_nonzero(~np.isnan(masked_field))))
    #print(f"data all shape is {data_all.shape}")
	#Retrain the non-masked elements and store in data_all
    grid_counter = 0


    # stack into 1d array except for nans
    # for 2D fields
    if (config.level_zonal_bottom != LevelZonalBottom.Three_D.value):
        for j in range(len(dim_j)):
            for i in range(len(dim_i)):
                if ~np.isnan(masked_field[j,i]): #if masked field is not a nan -> store data 
                    data_all[:, grid_counter]	= var_masked[:, j,i]
                    grid_counter += 1
        data_all = data_all
    # for 3D field
    elif (config.level_zonal_bottom == LevelZonalBottom.Three_D.value):
        for j in range(len(dim_j)):
            for i in range(len(dim_i)):
                for k in range(len(dim_k)):
                    if ~np.isnan(masked_field[k,j,i]): #if masked field is not a nan -> store data 
                        data_all[:, grid_counter]	= var_masked[:,k, j, i]
                        grid_counter += 1
        data_all = data_all
    #print(f'data_all = shape {data_all.shape}')
    
    
    # apply time reduction if > 1            
    if config.red_time > 1:
        print('Time reduction is initialised') 
        print('Time mean is taken over', config.red_time, 'unit time')
        
        time_2		= np.zeros(int(len(time) / config.red_time)) 
        data_all_2	= np.zeros((len(time_2), len(data_all[0,:])))
        
        for time_i in range(len(time_2)):
            #Take time mean to reduce the amount of time
            time_2[time_i]		= np.mean(time[time_i * config.red_time: (time_i + 1) * config.red_time]) 
            data_all_2[time_i,:]	= np.mean(data_all[time_i * config.red_time: (time_i + 1) * config.red_time,:], axis = 0) 
            
        time		= time_2
        data_all	= data_all_2
        print(f'data_all after time reduction = shape {data_all.shape}')

    elif config.red_time <1:
        return
    
    

    #print(f"data_all shape = {data_all.shape}")
    nancount = np.count_nonzero(np.isnan(data_all))
    infcount = np.count_nonzero(np.isinf(data_all))

    if nancount > 0:
        data_all[np.isnan(data_all) == True] = 0
        print("Replaced" + str(nancount) + " nans with 0...")
        print("inf = " + str(infcount))

    #print(data_all)

    data_all = DetrendProcedure_config(config=config, data_all=data_all, time=time)
    #Demean and normalise the data
    print("Substracting the mean and normalizing by std")
    data_all	= data_all - np.mean(data_all, axis = 0)
    data_std = np.std(data_all, axis = 0)

    data_all	= data_all / data_std


    if (config.level_zonal_bottom == LevelZonalBottom.Three_D.value): # would be better to change this., e.g, by putting dim_i, dim_j, and dim_k in a list and returning the list
        dims = [dim_i, dim_j, dim_k]
        print(dims)
    else:
        dims = [dim_i, dim_j]   

    return time, dims, data_all, data_std,masked_field	
	
