import pandas as pd 
import xarray as xr
import numpy as np
import sys
import os

sys.path.append("/Users/toonbense/Documents/GitHub/iLC_TOOLS") #path to iLC_TOOLS where TB_toolbox is located
from TB_toolbox import fix_lons_dataset_ilc_to_min180_180

import random




df12k = pd.read_csv('/Users/toonbense/Documents/Proxy_Data/Temp12k/T12K_AtlanticArctic_marine_Temp_unique_max100yrdt.csv')
df2k = pd.read_csv('/Users/toonbense/Documents/Proxy_Data/Temp2k/T2K_AtlanticArctic_marine_Temp_unique_max100yrdt.csv')

def retrieving_proxy_mask(proxy_cell_pct=100 , bool180E_180W = False):

    """
        Returns masked array with only grid cells that are located at SST proxy locations based on the above selections of data from the T2K and T12K databases.

        proxy_cell_pct: % of grid cells that are returned. 
        E.g., proxy_cell_pct = 50 returns a grid with a randomly selected half of the proxy_grid_cells masked as 1, the rest is zero.

        bool180E_180W: True, returns mask on -180 to 180 grid compatible with fix_lons_dataset_ilc_to_min180_180.
                        False, returns mask on 22.5 to 382.5 iLOVECLIM grid, compatible with post-processed ilc data.
        """

    if bool180E_180W == True: 
        ds_masks = fix_lons_dataset_ilc_to_min180_180(xr.open_dataset('/Users/toonbense/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/iLOVECLIM_code_TB/masks_ilc_tb.nc'))
    elif bool180E_180W == False:
        # must be on iLOVECLIm 22.5 - 382.5 lon grid, because Read_in_Data_iLC loads in data that way.
        ds_masks = (xr.open_dataset('/Users/toonbense/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/iLOVECLIM_code_TB/masks_ilc_tb.nc'))

    locations_list_of_lists = [] # store locations data from db into list of [lat,lon] lists
    for i in range(df12k.shape[0]):
        list_temp = []
        list_temp.append(df12k['geo_meanLat'].iloc(0)[i])
        list_temp.append(df12k['geo_meanLon'].iloc(0)[i])
        locations_list_of_lists.append(list_temp)

    for i in range(df2k.shape[0]):
        list_temp = []
        list_temp.append(df2k['geo_meanLat'].iloc(0)[i])
        list_temp.append(df2k['geo_meanLon'].iloc(0)[i])
        locations_list_of_lists.append(list_temp)

    unique_locations_list = [list(t) for t in set(tuple(lst) for lst in locations_list_of_lists)]     # remove duplicate locations
    print(f"unique proxy locations: {len(unique_locations_list)}")

    zeros = xr.zeros_like(ds_masks['mask_global_ocean']) # prepare xr gridded data where we fill locations with proxy grid with 1.
    lats = zeros.lat.data
    lons = zeros.lon.data
    proxy_mask = zeros.copy()
    
    checking_locs = []
    for i in range(len(unique_locations_list)):
        lon_proxy = unique_locations_list[i][1]

        if bool180E_180W == False: # adjust the lon proxy (-180 to 180) to the iLOVECLIM (22.5 to 382.5 grid)
            if ((lon_proxy > 23) &( lon_proxy < 180)):  
                lon_proxy_adjusted = lon_proxy
            elif((lon_proxy > -180) &( lon_proxy < 0)):
                lon_proxy_adjusted = 360 + lon_proxy # this lon_proxy is negative, thus + works.
            elif ((lon_proxy > 0) &( lon_proxy < 23)):
                lon_proxy_adjusted = 360 + lon_proxy
            lon_proxy = lon_proxy_adjusted
        
        loc = zeros.sel(lat = unique_locations_list [i][0], lon = lon_proxy, method='nearest')
        local_lat_i = np.argmin(np.abs(lats - loc.lat.data))
        locat_lon_i =np.argmin(np.abs(lons - loc.lon.data)) 
        checking_locs.append([local_lat_i, locat_lon_i])
        proxy_mask[local_lat_i, locat_lon_i] = 1 #contains latxlon array with 1s where there are proxy locations


    # select a percentage of masked grid cells
    if proxy_cell_pct == 100:
        proxy_mask = proxy_mask
        
    elif (proxy_cell_pct < 100) & (proxy_cell_pct > 1):
        unique_grid_cells = np.argwhere(proxy_mask.data == 1)
        cell_fraction = 1 - (proxy_cell_pct / 100) # randomly select pct of grid cells to be set to 0 
        n_to_select = int(cell_fraction*len(unique_grid_cells))
        set_to_zeros_index = random.sample(list(unique_grid_cells), n_to_select)
        for (i, j) in set_to_zeros_index: 
            proxy_mask[i, j] = 0 # now proxy_mask contains 1s in proxy_cell_pct of the amount of grid cells it would for a full (100% proxy grid cell) mask.
        print(f"iLOVECLIM unique locations in ocean mask: {proxy_mask.sum()}")
    else:
        print("Check proxy_cell_pct is between 1 and 100")
        return
    
    proxy_masked_ma = np.ma.masked_array(proxy_mask, mask=False) # return masked array to function similarly as a ds_masks[mask] opened through netcdf, required for readInIloveclim code.

    return proxy_masked_ma
    

def subsampling_data_all_std_masked_field(list_data_all, list_data_std,  masked_field, proxy_cell_pct):
    """ 
    Subsamples a certain % of masked locations based on output from ReadInData_iLOVECLIM.
    Intended use is in MonteCarlo Proxy_Field simulations, to prevent having to Read in the data all the time.

    Input:
    list_data_all:from Read_in_Data_iLC, which should have proxy_cell_pct = 100
    list_data_std:from Read_in_Data_iLC, which should have proxy_cell_pct = 100
    masked_field:from Read_in_Data_iLC, which should have proxy_cell_pct = 100

    proxy_cell_pct: The separate proxy_cell_pct to subsample the input data.

    Returns:
    
    Subsampled data_all, data_std, and masked field in the correct format for further mssa_runs.
    
    """

    if len(list_data_all) == 1:
        data_all = list_data_all[0]
        data_std = list_data_std[0]
    else:
         print("code must be changed to work with multiple variables")
         return

    channels = data_all.shape[1]

    list_channel_index = list(np.arange(0,channels,1))

    import random 
    if proxy_cell_pct == 100:
            data_all = data_all
            masked_field = masked_field
            data_std = data_std

    elif (proxy_cell_pct < 100) & (proxy_cell_pct > 1): # subsample data_all, masked_field, and data_std for every realizations..
        cell_fraction = (proxy_cell_pct / 100) # randomly select pct of grid cells
        n_to_select = int(cell_fraction*channels)
        subsampled_indexes = list(np.sort(random.sample(list_channel_index, n_to_select)))

        data_all = data_all[:,subsampled_indexes]
        data_std = data_std[subsampled_indexes]
        
        masked_field_indexes = np.argwhere(np.abs(masked_field) >0)
        masked_field_subsampled_indexes = masked_field_indexes[subsampled_indexes,:]
        masked_field_new = np.zeros_like(masked_field)
        for (i,j) in masked_field_subsampled_indexes:
            masked_field_new[i,j] = 1
        masked_field = masked_field_new
        print(masked_field.sum())

    else:
        print("Check proxy_cell_pct is between 1 and 100")

    if ((masked_field[0,0] == 0)):
        print(f"Ensuring masked field has nans and not zeros...")
        masked_field = np.where(masked_field == 0, np.nan, masked_field)
        masked_field = np.ma.masked_array(masked_field)   

    return [data_all], [data_std] ,  masked_field









def retrieving_unique_proxy_locations(proxy_cell_pct=100):
    """
    Returns list with SST proxy locations (lat,lon) based on the above selections of data from the T2K and T12K databases.

    proxy_cell_pct: % of grid cells that are returned. 
    E.g., proxy_cell_pct = 50 returns a list with SST proxy locations based on half of the unique proxy locations

    """

    locations_list_of_lists = []
    for i in range(df12k.shape[0]):
        list_temp = []
        list_temp.append(df12k['geo_meanLat'].iloc(0)[i])
        list_temp.append(df12k['geo_meanLon'].iloc(0)[i])
        locations_list_of_lists.append(list_temp)

    for i in range(df2k.shape[0]):
        list_temp = []
        list_temp.append(df2k['geo_meanLat'].iloc(0)[i])
        list_temp.append(df2k['geo_meanLon'].iloc(0)[i])
        locations_list_of_lists.append(list_temp)
    # remove duplicates 
    unique_locations_list = [list(t) for t in set(tuple(lst) for lst in locations_list_of_lists)]

    print(f" unique locations = {len(unique_locations_list)}")
    if (proxy_cell_pct < 100) & (proxy_cell_pct > 1):
        cell_fraction = proxy_cell_pct /100
        n_to_select = int(cell_fraction*len(unique_locations_list))
        selected = random.sample(unique_locations_list, n_to_select) #randomly select x% of locations from list

        unique_locations_list = selected

    elif proxy_cell_pct== 100:
        unique_locations_list = unique_locations_list
    else:
        print("Check proxy_cell_pct is between 1 and 100")
        return
    
    print(f"selected unique locations = {len(unique_locations_list)}")
    print(f"---------------------------------------------------")

    return unique_locations_list

    '''zeros = xr.zeros_like(ds_masks['mask_global_ocean'])
    lats = zeros.lat.data
    lons = zeros.lon.data

    proxy_mask = zeros.copy()

    for i in range(len(unique_locations_list)):
        lon_proxy = unique_locations_list[i][1]

        if bool180E_180W == False: # adjust the lon proxy (-180 to 180) to the iLOVECLIM (22.5 to 382.5 grid)
            if ((lon_proxy > 23) &( lon_proxy < 180)):  
                lon_proxy_adjusted = lon_proxy
            elif((lon_proxy > -180) &( lon_proxy < 0)):
                lon_proxy_adjusted = 360 + lon_proxy # this lon_proxy is negative, thus + works.
            elif ((lon_proxy > 0) &( lon_proxy < 23)):
                lon_proxy_adjusted = 360 + lon_proxy
            lon_proxy = lon_proxy_adjusted
        

        loc = zeros.sel(lat = unique_locations_list [i][0], lon = lon_proxy, method='nearest')
        local_lat_i = np.argmin(np.abs(lats - loc.lat.data))
        locat_lon_i =np.argmin(np.abs(lons - loc.lon.data)) 

        proxy_mask[local_lat_i, locat_lon_i] = 1

    proxy_masked_ma = np.ma.masked_array(proxy_mask, mask=False)

    return proxy_masked_ma'''