import numpy as np
import xarray as xr
import netCDF4 as netcdf

from mssakit.MSSA_config import MSSAConfig
from mssakit.MSSA_config import LevelZonalBottom





# MSSA functions
def retrieve_RC(config: MSSAConfig, EEOFs_pair, RC_sOR_PCs_or_STDs="RCs", zscore = True):
    """ Retrieve the full spatio-temporal fields of an EEOF pair on the iLOVECLIM grid.
    
    Input:
    config: MSSAConfig file \n
    EEOFs_pair :list # the significant EEOF pair to retrieve. Input = the EEOF number, not the python index!
    RC_sOR_PCs_or_STDs: str - either retrieve RCs or PCs or STD
    zscore: True or False, if True, returns RC in z-score, if False returns RC in original variable unit.

    Returns:
    list_xr_data_arrays : list of xr_data_arrays, 1 for each variable used in the mssa run.

    
    """
    EEOFs_pair_index = [] # create list with the index of the EEOF pair to select data with.
    EEOFs_pair_index.append(EEOFs_pair[0] -1 )
    EEOFs_pair_index.append(EEOFs_pair[1] -1 )


    filename = config.get_filename()


    EEOF_data = netcdf.Dataset(config.output_directory+'EEOFs/'+filename+'.nc', mode='r')    
    time        	= EEOF_data.variables['time'][:]		
    #
    if ((config.level_zonal_bottom == LevelZonalBottom.LEVEL.value) | (config.level_zonal_bottom == LevelZonalBottom.BOTTOM.value)| (config.level_zonal_bottom == LevelZonalBottom.VOL.value)):
        lon         	= EEOF_data.variables['lon'][:]
    elif (config.level_zonal_bottom == LevelZonalBottom.ZONAL.value):
        depth         	= EEOF_data.variables['depth'][:] 		
    elif (config.level_zonal_bottom == LevelZonalBottom.Three_D.value):
        lon         	= EEOF_data.variables['lon'][:]
        depth         	= EEOF_data.variables['depth'][:] 		

        
    lat	        = EEOF_data.variables['lat'][:]
    masked_field	= EEOF_data.variables['masked_field'][:]
    print(f"shape masked field = {masked_field.shape}")
    '''ST_PCs = EEOF_data.variables['ST_PCs'][:]
    var_expl_mssa = EEOF_data.variables['var_expl_mssa'][:]
    PCs = EEOF_data.variables['ST_PCs']'''


    RCs_list = []
    STD_list = []

    for i in range(config.numberOfVar): 
        RC_var_name_String = 'RCs_' + str(i+1)
        RC = EEOF_data.variables[RC_var_name_String][:]
        RCs_list.append(RC)         #RCs_list contains a list of arrays, where each array represents a 3D array of shape (time, channel, EEOF_number) (where EEOF_number = n_rec = 50)

    
        std_var_name_String = 'std_var' + str(i+1)
        STD = EEOF_data.variables[std_var_name_String][:]
        STD_list.append(STD)        #STD_list contains a list of arrays, where each array represents a 1D array of shape (channel)


    if RC_sOR_PCs_or_STDs == "PCs":
        return EEOF_data.variables['ST_PCs'][:,EEOFs_pair_index].data


    '''    Spectra_data = netcdf.Dataset(config.output_directory+'MSSA_Spectra/'+config.get_filename() +'.nc', mode='r')
    freq         = Spectra_data.variables['freq'][:].data   / red_time #adjust for time reduction.
    #MC_power	    = Spectra_data.variables['MC_power'][:].data
    data_power 		= Spectra_data.variables['data_power'][:].data
    T = Spectra_data.variables['T'][:].data * red_time
    sigma_T = Spectra_data.variables['sigma_T'][:].data * red_time'''

    def retrieve_RC_per_var(config: MSSAConfig, EEOFs_pair_index, RC_array, STD_array, RC_sOR_PCs_or_STDs, masked_field = masked_field):
        RCs = RC_array
        print(RCs.shape)
        print(RCs[[0,1]].shape)
        RC_mode = np.sum(RCs[:,:,EEOFs_pair_index], axis=2)   # sum RC1 and RC2.

        if zscore == False: # Remap the RC to the unit of the original variable if zscore = False
            RC_mode = RC_mode * STD_array
        if RC_sOR_PCs_or_STDs == 'STD':
            zero_array = np.zeros_like(RC_mode) # 
            ones_array = np.where(zero_array == 0,1,np.nan)
            RC_mode = ones_array * STD_array

        nans = np.empty(len(time),)  #create an array of nans to fill cells with no ocean values (!) in shape of (time, 0)
        nans[:] = np.nan
        if ((config.level_zonal_bottom == LevelZonalBottom.LEVEL.value) | (config.level_zonal_bottom == LevelZonalBottom.BOTTOM.value)| (config.level_zonal_bottom == LevelZonalBottom.VOL.value)):
                dim_j = lat; dim_i = lon
        elif (config.level_zonal_bottom == LevelZonalBottom.ZONAL.value):
            dim_j = depth; dim_i = lat
        elif (config.level_zonal_bottom == LevelZonalBottom.Three_D.value):
            dim_j = lat; dim_i = lon; dim_k=depth
        
        counter = 0
        if (config.level_zonal_bottom != LevelZonalBottom.Three_D.value):
            Eofmap = np.zeros((len(time),masked_field.shape[0], masked_field.shape[1] ))  #create an array of zeros to fill in the shape of (time, lat, lon)
            #return masked_field
            if ((masked_field[0,0] == 0) & (config.proxy_cell_pct != 100)):
                print(f"Temporarily fixing masked field (shape {masked_field.shape} for configs with subsampled proxy field")
                masked_field = np.where(masked_field == 0, np.nan, masked_field)
                masked_field = np.ma.masked_array(masked_field)

            for j in range(len(dim_j)):
                for i in range(len(dim_i)):
                    if  ~np.isnan(masked_field[j,i]):
                        Eofmap[:, j, i] = RC_mode[:, counter] 
                        counter +=1

                    else:
                        Eofmap[:, j, i] = nans
            Spacetime = Eofmap

        elif (config.level_zonal_bottom == LevelZonalBottom.Three_D.value):

            Eofmap = np.zeros((len(time),masked_field.shape[0], masked_field.shape[1], masked_field.shape[2] ))  #create an array of zeros to fill in the shape of (time, lat, lon)

            for j in range(len(dim_j)):
                for i in range(len(dim_i)):
                    for k in range(len(dim_k)):
                        if  ~np.isnan(masked_field[k,j,i]):
                            Eofmap[:,k, j, i,] = RC_mode[:, counter] 
                            counter +=1

                        else:
                            Eofmap[:,k, j, i] = nans
            Spacetime = Eofmap

        
          
        if ((config.level_zonal_bottom == LevelZonalBottom.LEVEL.value) | (config.level_zonal_bottom == LevelZonalBottom.BOTTOM.value)| (config.level_zonal_bottom == LevelZonalBottom.VOL.value)):
            ### reorders lon dimension to go from -180 to 180 instead of -22.5 to something... consdider moving to other section in code, maybe even to readinData
            dim_i_fixed = np.concatenate([ dim_i[52:-8],dim_i[-8:] , dim_i[:52]])
            Spacetime_fixed = np.concatenate([Spacetime[:,:, 52:-8] , Spacetime[:,:, -8:] ,Spacetime[:,:,:52] ], axis = 2)
            da_var1 = xr.DataArray(data=Spacetime_fixed,
                        dims=('time', 'lat', 'lon'),
                        coords={'time': time,'lat': dim_j,'lon': dim_i_fixed})
            
        elif config.level_zonal_bottom == LevelZonalBottom.ZONAL.value:
            da_var1 = xr.DataArray(data=Spacetime,
                        dims=('time', 'level', 'lat'),
                        coords={'time': time,'level': dim_j,'lat': dim_i})
            
        elif config.level_zonal_bottom == LevelZonalBottom.Three_D.value:
            dim_i_fixed = np.concatenate([ dim_i[52:-8],dim_i[-8:] , dim_i[:52]])
            Spacetime_fixed = np.concatenate([Spacetime[:,:,:, 52:-8] , Spacetime[:,:,:, -8:] ,Spacetime[:,:,:,:52] ], axis = 3)
            da_var1 = xr.DataArray(data=Spacetime_fixed,
                        dims=('time', 'level', 'lat', 'lon'),
                        coords={'time': time, 'level':depth,'lat': dim_j,'lon': dim_i_fixed})
    

        return da_var1
    


    list_xr_data_arrays = []
    for i in range(config.numberOfVar):
        da = retrieve_RC_per_var(config, EEOFs_pair_index, RCs_list[i], STD_list[i], RC_sOR_PCs_or_STDs=RC_sOR_PCs_or_STDs, masked_field=masked_field)
        list_xr_data_arrays.append(da)

    max_number_of_var = 5
    remaining = max_number_of_var-config.numberOfVar
    for i in range(remaining):
        list_xr_data_arrays.append(None)
        
    return list_xr_data_arrays

def retrieve_gridded_std(config: MSSAConfig):
    filename = config.get_filename()


    EEOF_data = netcdf.Dataset(config.output_directory+'EEOFs/'+filename+'.nc', mode='r')    
    time        	= EEOF_data.variables['time'][:]		
    #
    if ((config.level_zonal_bottom == LevelZonalBottom.LEVEL.value) | (config.level_zonal_bottom == LevelZonalBottom.BOTTOM.value)| (config.level_zonal_bottom == LevelZonalBottom.VOL.value)):
        lon         	= EEOF_data.variables['lon'][:]
    elif (config.level_zonal_bottom == LevelZonalBottom.ZONAL.value):
        depth         	= EEOF_data.variables['depth'][:] 		
    elif (config.level_zonal_bottom == LevelZonalBottom.Three_D.value):
        lon         	= EEOF_data.variables['lon'][:]
        depth         	= EEOF_data.variables['depth'][:] 		

        
    lat	        = EEOF_data.variables['lat'][:]
    masked_field	= EEOF_data.variables['masked_field'][:]
    RCs_list = []
    STD_list = []

    for i in range(config.numberOfVar): 
        RC_var_name_String = 'RCs_' + str(i+1)
        RC = EEOF_data.variables[RC_var_name_String][:]
        RCs_list.append(RC)         #RCs_list contains a list of arrays, where each array represents a 3D array of shape (time, channel, EEOF_number) (where EEOF_number = n_rec = 50)

    
        std_var_name_String = 'std_var' + str(i+1)
        STD = EEOF_data.variables[std_var_name_String][:]
        STD_list.append(STD)        #STD_list contains a list of arrays, where each array represents a 1D array of shape (channel)



   