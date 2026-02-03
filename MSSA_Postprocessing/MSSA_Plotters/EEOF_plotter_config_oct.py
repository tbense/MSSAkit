import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import netCDF4 as netcdf	
import cartopy.crs as ccrs
import cmocean
import xarray as xr
import warnings



from mssakit.MSSA_config import MSSAConfig
from mssakit.MSSA_config import LevelZonalBottom

from MSSA_Postprocessing.Retrieve_PC_RC import retrieve_RC
import sys
import os
sys.path.append("/Users/toonbense/Documents/GitHub/iLC_TOOLS") # Path to iLC_TOOLS where TB_toolbox is located
from TB_toolbox import horiz_weighted_averaging
from TB_toolbox import fix_lons_dataset_ilc_to_min180_180
from TB_toolbox import add_map_features
from TB_toolbox import polarCentral_set_latlim
from TB_toolbox import lead_lag_cor



def EEOF_plotter_config(config: MSSAConfig, EEOFs_pair, start_t_spatial, step_size_spatial, start_time_Hovm, end_time_Hovm, hovm_v_max_abs = 1, zscore=True):
    """
    Plots several standard plots and stores them in a pdf for a given EEOF pair and specific MSSA_config.

    Input:
    config :    MSSAConfig file \n
    EEOFs_pair: list, 2 EEOFs that form a significant pair. Put the number of the EEOF, NOT the index. 
    start_t_spatial: int,  start time for spatio-temporal plots.
    step_size_spatial: int, step size for spatio-temporal plots 
    start_time_Hovm: int, start time for hovmollers
    end_time_Hovm: int, end time for hovmollers

    Output:
    pdf stored in  config.output_directory + "Figures/" + 'Quad_Hof_etc' + config.get_string_file_info() + "STPC_pairs_" + str(EEOFs_pair[0]) + str(EEOFs_pair[1]) + ".pdf"
    
    """



    EEOFs_pair_index = [] # create list with the index of the EEOF pair to select data with.
    EEOFs_pair_index.append(EEOFs_pair[0] -1 )
    EEOFs_pair_index.append(EEOFs_pair[1] -1 )

    def plot_Nearside_lat_lon(config : MSSAConfig, da, var_name, maxnum=4):
            # plot Arctic+Atlantic with ccrs.NearsidePerspective  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            fig, axes = plt.subplots(2, maxnum,figsize=(12,8),layout = 'tight', subplot_kw={'projection': ccrs.NearsidePerspective(central_longitude=-40, central_latitude=50)})   #
            count_loc = int(start_t_spatial / config.red_time)

            for i in range(1, 2 * maxnum):
                ax = axes[(i - 1) // maxnum, (i - 1) % maxnum]  # Access the correct subplot

                mesh = da.isel(time=count_loc).plot(ax=ax, cmap=cmocean.cm.balance,transform=ccrs.PlateCarree(), add_colorbar=False)
                add_map_features(ax)
                ax.set_title(str(count_loc * config.red_time))
                count_loc += int(step_size_spatial / config.red_time)
                warnings.filterwarnings('ignore')

            axes[1,-1].set_visible(False)

            cbar = fig.colorbar(mesh, ax=axes[1,-1], shrink = 0.6, location = 'right', aspect=10, fraction = 0.8, label = cbar_label[var_name])
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.001)
            plt.title(f'{config.label} {config.run_name} {config.level_zonal_bottom} {var_name}')

            pdf.savefig(fig)
    def plot_Arctic_stero_lat_lon(config : MSSAConfig, da, var_name, maxnum=4):
        # plot Arctic region only NorthPolarStereo ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        lat_lims = [60,90]

        fig, axes = plt.subplots(2, maxnum,figsize=(12,8),layout = 'tight', subplot_kw={'projection': ccrs.NorthPolarStereo()})   #
        count_loc = int(start_t_spatial / config.red_time)

        for i in range(1, 2 * maxnum):
            ax = axes[(i - 1) // maxnum, (i - 1) % maxnum]  # Access the correct subplot

            mesh = da.isel(time=count_loc).where(da['lat']>lat_lims[0]).plot(ax=ax, cmap=cmocean.cm.balance,transform=ccrs.PlateCarree(), add_colorbar=False)
            polarCentral_set_latlim(lat_lims, ax)
            add_map_features(ax)
            ax.set_title(str(count_loc * config.red_time))
            count_loc += int(step_size_spatial / config.red_time)
            warnings.filterwarnings('ignore')

        axes[1,-1].set_visible(False)
        cbar = fig.colorbar(mesh, ax=axes[1,-1], shrink = 0.6, location = 'right', aspect=10, fraction = 0.8, label = cbar_label[var_name])
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.001)

        plt.title(f'{config.label} {config.run_name} {config.level_zonal_bottom} {var_name}')

        pdf.savefig(fig)

    def plot_Atlantic_lat_lon(config : MSSAConfig, da, var_name, maxnum=4):
        # plot Atlantic central with limited arctic - ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # probably not pretty with Atlantic only, but probably will not use atlantic only anyways.
        # plot Atlantic maps

        count_loc = int(start_t_spatial / config.red_time)

        # lat x lon plot------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        fig, axes = plt.subplots(2, maxnum,figsize=(12,8),layout = 'tight', subplot_kw={'projection': ccrs.Robinson()})   # Create a figure with subplots
        lon_lims = [-85,15]

        for i in range(1, 2 * maxnum):
            ax = axes[(i - 1) // maxnum, (i - 1) % maxnum]  # Access the correct subplot
            mesh = da.isel(time=count_loc).plot(ax=ax, cmap=cmocean.cm.balance,transform=ccrs.PlateCarree(), add_colorbar=False)
            ax.set_extent([lon_lims[0],lon_lims[1], -30, 90], ccrs.PlateCarree())
            #add_map_features(ax)

            gl = ax.gridlines(draw_labels=False,linestyle='-',alpha=0.8,linewidth=0.8, transform = ccrs.PlateCarree())
            gl.xlocator = mticker.FixedLocator([-60, -30, -0, 30, 60])
            gl.ylocator = mticker.FixedLocator([-30, 0,30, 60])
            if i%maxnum ==1:
                gl.left_labels = True
            if i>maxnum:
                gl.bottom_labels = True
            
            ax.set_title(str(count_loc * config.red_time))

            count_loc += int(step_size_spatial / config.red_time)
            warnings.filterwarnings('ignore')
        axes[1,-1].set_visible(False)
        # Create and set up the colorbar
        cbar = fig.colorbar(mesh, ax=axes[1,-1], shrink = 0.9, location = 'right', aspect = 10, fraction = 0.8, label = cbar_label[var_name])
        plt.title(f'{config.label} {config.run_name} {config.level_zonal_bottom} {var_name}')


        pdf.savefig(fig)

    def plot_global_ocean_latlon(config : MSSAConfig, da, var_name, maxnum=4):
        # plot Atlantic central with limited arctic - ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # probably not pretty with Atlantic only, but probably will not use atlantic only anyways.
        # plot Atlantic maps

        count_loc = int(start_t_spatial / config.red_time)

        # lat x lon plot------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        fig, axes = plt.subplots(2, maxnum,figsize=(12,8),layout = 'tight', subplot_kw={'projection': ccrs.Robinson()})   # Create a figure with subplots
        lon_lims = [-85,15]

        for i in range(1, 2 * maxnum):
            ax = axes[(i - 1) // maxnum, (i - 1) % maxnum]  # Access the correct subplot
            mesh = da.isel(time=count_loc).plot(ax=ax, cmap=cmocean.cm.balance,transform=ccrs.PlateCarree(), add_colorbar=False)
            #ax.set_extent([lon_lims[0],lon_lims[1], -30, 90], ccrs.PlateCarree())
            #add_map_features(ax)

            gl = ax.gridlines(draw_labels=False,linestyle='-',alpha=0.8,linewidth=0.8, transform = ccrs.PlateCarree())
            #gl.xlocator = mticker.FixedLocator([-60, -30, -0, 30, 60])
            #gl.ylocator = mticker.FixedLocator([-30, 0,30, 60])
            if i%maxnum ==1:
                gl.left_labels = True
            if i>maxnum:
                gl.bottom_labels = True
            
            ax.set_title(str(count_loc * config.red_time))

            count_loc += int(step_size_spatial / config.red_time)
            warnings.filterwarnings('ignore')
        axes[1,-1].set_visible(False)
        # Create and set up the colorbar
        cbar = fig.colorbar(mesh, ax=axes[1,-1], shrink = 0.5, location = 'right', aspect = 10, fraction = 0.8, label = cbar_label[var_name])
        plt.title(f'{config.label} {config.run_name} {config.level_zonal_bottom} {var_name}')


        pdf.savefig(fig)


    def plot_Atlantic_zonal(config : MSSAConfig, da, var_name, maxnum=4):
        # lat x depth plot ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        fig, axes = plt.subplots(2, maxnum, figsize=(12,8), layout='tight')
        count_loc = int(start_t_spatial / config.red_time)

        levels = np.arange(-1, 1.1, 0.1)
        for i in range(1, 2 * maxnum):
            ax = axes[(i - 1) // maxnum, (i - 1) % maxnum]  # Access the correct subplot

            mesh = da.isel(time=count_loc).plot(ax= ax, cmap=cmocean.cm.balance, add_colorbar=False)
            ax.set_title(str(start_t_spatial + count_loc * config.red_time))
            count_loc += int(step_size_spatial / config.red_time)
            ax.grid(alpha =0.8, lw = 0.8,ls = 'dashed')
            ax.set_xlim((-30,90))
            if i%maxnum ==1:
                ax.set_yticks([0,-1000,-2000,-3000,-4000,-5000])
                ax.set_yticklabels(['0', '1', '2', '3', '4', '5'])
                ax.set_ylabel('Depth (km)')
            else:
                ax.set_yticks([0,-1000,-2000,-3000,-4000,-5000])
                ax.set_yticklabels([])

            if i>maxnum:
                ax.set_xticks([-30,0,30,60,90])
                ax.set_xlabel("Latitude (\u00b0N)")
            else:
                ax.set_xticks([-30,0,30,60,90])
                ax.set_xticklabels([])      
    
        axes[1,-1].set_visible(False)
        # Create and set up the colorbar
        cbar = fig.colorbar(mesh, ax=axes[1,-1], shrink = 0.9, location = 'right', aspect = 10, fraction = 0.8, label = cbar_label[var_name])

        plt.title(f'{config.label} {config.run_name} {config.level_zonal_bottom} {var_name}')

        pdf.savefig(fig)

    def plot_hovm_surf_zonal(config : MSSAConfig, da, var_name):
        fig, ax = plt.subplots()  # Use subplots instead of subplot
                
        hov_zonal = da.mean(dim='lon', skipna=True).transpose().plot(add_colorbar=False, vmin = -hovm_v_max_abs, vmax = hovm_v_max_abs, cmap = 'Spectral_r')              # could still adjsut to include varying area/volume better
        cbar = fig.colorbar(hov_zonal, ax=ax, shrink = 0.85, location = 'right', aspect=15, fraction = 0.05, label = cbar_label[var_name])
        
        ax.set_ylim((-30,90))
        ax.set_xlim(start_time_Hovm,  end_time_Hovm)

        ax.set_ylabel("Latitude (\u00b0N)")
        ax.set_xlabel("Time (years)")
        ax.set_title("Hovmöller diagram - zonal mean")

        pdf.savefig(fig)
        
    def plot_hovm_surf_merid_40_60N(config : MSSAConfig, da, var_name):

        fig, ax = plt.subplots()  # Use subplots instead of subplot
                
        hov_40_60 = da.isel(lat=slice(65,75)).mean(dim='lat', skipna=True).plot(add_colorbar=False, vmin = -hovm_v_max_abs, vmax = hovm_v_max_abs, cmap = 'Spectral_r')      # could still adjsut to include varying area/volume better
        cbar = fig.colorbar(hov_40_60, ax=ax, shrink = 0.85, location = 'right', aspect=15, fraction = 0.05, label = cbar_label[var_name])
        
        ax.set_xlabel("Longitude (\u00b0E)")
        ax.set_ylabel("Time (years)")
        ax.set_title("Hovmöller diagram - meridional (40N-60N) mean ")
        ax.set_xlim((-60,-10))
        ax.set_ylim(start_time_Hovm,  end_time_Hovm)

        pdf.savefig(fig)

    def plot_hovm_zonal_250_1000m(config : MSSAConfig, da, var_name):
        
        fig, ax = plt.subplots()  # Use subplots instead of subplot
                
        bool_250_1000 = ((da.level > -1000) & ( da.level < -250))
        slice_var_1 = da.where(bool_250_1000)

        v1 = ds_masks.Volume*ds_masks.mask_atl_arc
        v2 = v1.sum(dim='lon') # gives total zonal volume per grid cell in latxlon
        v3 = v2.where(bool_250_1000)

        hov_depth_250_1000 = (slice_var_1 * v3 / v3.sum(dim='level')).sum(dim=('level')).transpose().plot(add_colorbar=False, vmin = -hovm_v_max_abs, vmax = hovm_v_max_abs, cmap = 'Spectral_r')     # zonal volume weighted average..
        #hov_depth_250_1000 = da.where(bool_250_1000).mean(dim='level', skipna=True).transpose().plot(add_colorbar=False, vmin = -hovm_v_max_abs, vmax = hovm_v_max_abs, cmap = 'Spectral_r')  # could still adjsut to include varying area/volume better
        cbar = fig.colorbar(hov_depth_250_1000, ax=ax, shrink = 0.85, location = 'right', aspect=15, fraction = 0.05, label = cbar_label[var_name])

        ax.set_ylabel("Latitude (\u00b0N)")
        ax.set_xlabel("Time (years)")
        ax.set_title("Hovmöller diagram - lat (250-1000m depth) v time")



        ax.set_ylim((-30,90))
        ax.set_xlim(start_time_Hovm,  end_time_Hovm)

        pdf.savefig(fig)
    def plot_hovm_zonal_2000_3000m(config : MSSAConfig, da, var_name):

        fig, ax = plt.subplots()  # Use subplots instead of subplot
                       
        bool_2000_3000 = ((da.level > -3000) & ( da.level < -2000))
        slice_var_1 = da.where(bool_2000_3000)

        v1 = ds_masks.Volume*ds_masks.mask_atl_arc
        v2 = v1.sum(dim='lon') # gives total zonal volume per grid cell in latxlon
        v3 = v2.where(bool_2000_3000)

        hov_depth_2000_3000 = (slice_var_1 * v3 / v3.sum(dim='level')).sum(dim=('level')).transpose().plot(add_colorbar=False, vmin = -hovm_v_max_abs, vmax = hovm_v_max_abs, cmap = 'Spectral_r')     # zonal volume weighted average...
        #hov_depth_1500_2500 = da.where(bool_1500_2500).mean(dim='level', skipna=True).transpose().plot(add_colorbar=False, vmin = -hovm_v_max_abs, vmax = hovm_v_max_abs, cmap = 'Spectral_r')  # could still adjsut to include varying area/volume better
        cbar = fig.colorbar(hov_depth_2000_3000, ax=ax, shrink = 0.85, location = 'right', aspect=15, fraction = 0.05, label = cbar_label[var_name])

        ax.set_ylabel("Latitude (\u00b0N)")
        ax.set_xlabel("Time (years)")
        ax.set_title("Hovmöller diagram - lat (2000-3000m depth) v time")

        ax.set_ylim((-30,90))
        ax.set_xlim(start_time_Hovm,  end_time_Hovm)

        pdf.savefig(fig)
        
    def lead_lag_40_60N_250_1000m(config:MSSAConfig, da_var1, da_var2, ax1, ax2, var_name ):
        # calculate timeseries of weighted average over 40-60N and 250-1000m depth
        # calculate and plot lead-lag relation - clearly indicate what lags what (!)

        bool_depth = ((da_var1.lat > 40) & ( da_var1.lat < 60) & (da_var1.level < -250) & ( da_var1.level > -1000))
        
        
        slice_var_1 = da_var1.where(bool_depth)
        slice_var_2 = da_var2.where(bool_depth)

        v1 = ds_masks.Volume*ds_masks.mask_atl_arc
        v2 = v1.sum(dim='lon') # gives total zonal volume per grid cell in latxlon?
        v3 = v2.where(bool_depth)
        ts_var1_d = (slice_var_1 * v3 / v3.sum()).sum(dim=('lat','level'))
        ts_var2_d = (slice_var_2 * v3 / v3.sum()).sum(dim=('lat','level'))

        ax1.plot(ts_var1_d, label = variables_list[0])
        ax1.plot(ts_var2_d, label = var_name)
        ax1.legend()

        window = int(T[EEOFs_pair_index[0]]/config.red_time)
        lags_d, correlations_d = lead_lag_cor(ts_var1_d, ts_var2_d, dt=config.red_time, windowsize=  window)
        ax2.plot(lags_d, correlations_d,  label=var_name)

        ax2.axvline(x=0,color='k',ls='dashed')
        ax2.axhline(y=0,color='k',ls='dashed')

        ax2.set_xlabel('Lag (years)')
        ax2.set_ylabel("Correlation coefficient")
        ax2.set_ylim([-1, 1])
        ax2.set_xlim(-np.around(T[EEOFs_pair_index[0]]), np.around(T[EEOFs_pair_index[0]]))
        ax2.xaxis.set_major_locator(mticker.MultipleLocator(np.around(T[EEOFs_pair_index[0]]) / 4))

        ax2.grid()
        ax2.set_title("lead - lag cor. 40N-60N, 250-1000m - at lag <0 " + variables_list[0] + " leads the other variables")
        ax2.legend()

    #variable names and associated label

    variables_list = [config.variable,config.variable2, config.variable3, config.variable4, config.variable5 ]
    
    
    if zscore == True:
        cbar_label = {'salt':'Salinity (z-score)',
                'temp':'Temperature (z-score)',
                'u':' u (z-score)',
                'v':' v (z-score)',
                'rho':'Density (z-score)',
                'speed':'Flow speed (z-score)',
                'd13C':r'$\delta^{13}C$ (z-score)',
                'PaTh':'Pa/Th (z-score)',
                'mld' : 'MLD (z-score)'}
    else:
        cbar_label = {'salt':'Salinity (psu)',
                'temp':'Temperature (°C)',
                'u':' u (m/s)',
                'v':' v (m/s)',
                'rho':'Density (kg/$m^3$)',
                'speed':'Flow speed (m/s)',
                'd13C':r'$\delta^{13}C (‰)$',
                'PaTh':'Pa/Th',
                'mld' : 'MLD (m)'}

    ### ------------------------------------------------------------------------------------------------
    #load masks
    ds_masks  = fix_lons_dataset_ilc_to_min180_180(xr.open_dataset('/Users/toonbense/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/PhD/iLOVECLIM_code_TB/masks_ilc_tb.nc'))
    ### ------------------------------------------------------------------------------------------------
    #load data
    pdf_filename = config.output_directory + "Figures/" + 'Quad_Hof_etc' + config.get_string_file_info() + "STPC_pairs_" + str(EEOFs_pair[0]) + str(EEOFs_pair[1]) + ".pdf"

    EEOF_path = config.output_directory+'EEOFs/'+config.get_filename()+'.nc'
    EEOF_data = netcdf.Dataset(EEOF_path, mode='r')    
    var_expl_mssa = EEOF_data.variables['var_expl_mssa'][:]
    PCs = EEOF_data.variables['ST_PCs']

    Spectra_data = netcdf.Dataset(config.output_directory+'MSSA_Spectra/'+config.get_filename() +'.nc', mode='r')
    T = Spectra_data.variables['T'][:].data * config.red_time

    print('Variance captured by RCs is: ' + str(np.around(var_expl_mssa[EEOFs_pair[0]] +  var_expl_mssa[EEOFs_pair[1]],2) ) + " %")

    ### ------------------------------------------------------------------------------------------------
    #PLOTTING
    with PdfPages(pdf_filename) as pdf:

        ######PC vs time ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        timeaxis_PC = np.linspace(config.start_time+config.window_size/2,config.end_time-config.window_size,PCs.shape[0])
            
        fig, ax	= plt.subplots()
        plt.plot(timeaxis_PC,PCs[:,EEOFs_pair_index[0]], label='EEOF'+str(EEOFs_pair[0]))
        plt.plot(timeaxis_PC,PCs[:,EEOFs_pair_index[1]], label='EEOF'+str(EEOFs_pair[1]))

        plt.xlabel('Time ('+config.period_str+')')
        plt.ylabel('Amplitude')
        ax.grid()
        plt.legend()
        plt.title(f"{config.run_name} - {(np.around(var_expl_mssa[EEOFs_pair[0]] +  var_expl_mssa[EEOFs_pair[1]],2))}  %")

        #show()
        pdf.savefig(fig)
        plt.close()

        # lag-lead correlation ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        fig, ax	= plt.subplots()
                
        lag, correlation = lead_lag_cor(PCs[:,EEOFs_pair_index[0]], PCs[:,EEOFs_pair_index[1]],dt=config.red_time, windowsize=config.window_size)

        plt.axvline(x=0,color='k',ls='dashed')
        plt.axhline(y=0,color='k',ls='dashed')

        plt.plot(lag, correlation, '-r', linewidth = 2.0,label ='pair '+str(EEOFs_pair[0])+','+str(EEOFs_pair[1]))

        plt.xlabel('Lag ('+config.period_str+')')
        plt.ylabel("Correlation coefficient")
        plt.ylim([-1, 1])
        plt.xlim(-np.around(T[EEOFs_pair_index[0]]), np.around(T[EEOFs_pair_index[0]]))
        ax.xaxis.set_major_locator(mticker.MultipleLocator(np.around(T[EEOFs_pair_index[0]]) / 4))

        ax.grid()
        plt.title("lead - lag correlation")
        #show()
        pdf.savefig(fig)
        plt.close()  

        # periodogram ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        fig, ax = plt.subplots()

        def compute_fft_power_and_freq(series):
            fft_result = np.fft.fft(series)
            power = (np.real(fft_result)**2.0) + (np.imag(fft_result)**2.0)
            freq = np.fft.fftfreq(len(series))
            return power[:len(freq)//2], freq[:len(freq)//2]

        freq_series_1, freq_1 = compute_fft_power_and_freq(PCs[:, EEOFs_pair_index[0]])
        freq_series_2, freq_2 = compute_fft_power_and_freq(PCs[:, EEOFs_pair_index[1]])

        # plot
        ax.plot(freq_1/config.red_time, freq_series_1, '.-', label = 'EEOF'+str(EEOFs_pair[0]))
        ax.plot(freq_2/config.red_time, freq_series_2, '--', label = 'EEOF'+str(EEOFs_pair[1]))
        ax.legend()
        ax.set_xlabel('Frequency (1/'+config.period_str+')')
        ax.set_ylabel('Power')
        ax.set_xlim([0,0.02])
        ax.grid()
        g1 = ax.grid(visible=True, which='major', color='k', linestyle='-', linewidth=0.5)
        g2 = ax.grid(visible=True, which='minor', color='k', linestyle='-', linewidth=0.2)
        ax.minorticks_on() 
        plt.title('Periodogram')

        pdf.savefig(fig)
        plt.close()


        #SPATIOTEMPORAL------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ## retrieve xarray_da of EEOF pairs in a list

        list_xr_data_array_unfixed = (retrieve_RC(config=config, EEOFs_pair=EEOFs_pair, RCsOR_PCs="RCs", zscore=zscore))  # retrieves a da with dims timexlatxlon or timexlevelxlat (!)    
        list_xr_data_arrays = []
        for i in range(config.numberOfVar):
            list_xr_data_arrays.append(fix_lons_dataset_ilc_to_min180_180(list_xr_data_array_unfixed[i]))
                                      
        
        #### spatial plots that are plotted separately for each variable...
        for i in range(config.numberOfVar): # loop
            var_name = variables_list[i]
            da = list_xr_data_arrays[i]
            print("HERE" + var_name)
            if zscore == False:
                hovm_v_max_abs = np.around(da.std(dim='time').max(),2) # if original variable units are used, scale hovmoller to max std of lat x lon field?

            if ((config.level_zonal_bottom == LevelZonalBottom.LEVEL.value) |(config.level_zonal_bottom == LevelZonalBottom.VOL.value)):

                ### Plot Arctic
                if ((config.sel_region == 'Atlantic_Arctic') | (config.sel_region == 'Arctic')):
                    plot_Nearside_lat_lon(config, da=da, var_name=var_name) 
                    plt.close()

                    plot_Arctic_stero_lat_lon(config, da=da, var_name=var_name)
                    plt.close()


                ### Plot Atlantic
                if (config.sel_region == 'Atlantic_Arctic') | (config.sel_region == 'Atlantic'):
                    plot_Atlantic_lat_lon(config, da=da, var_name=var_name) 
                    plt.close()


                    #Hovmollers
                    plot_hovm_surf_zonal(config, da=da, var_name=var_name)
                    plt.close()
                    
                    plot_hovm_surf_merid_40_60N(config, da=da, var_name=var_name)
                    plt.close()

                if (config.sel_region == 'Global_Ocean'):
                    #plot global ocean
                    plot_global_ocean_latlon(config, da=da, var_name=var_name) 
                    plt.close()
                    # also plot atlantic
                    plot_Atlantic_lat_lon(config, da=da, var_name=var_name) 
                    plt.close()


            elif config.level_zonal_bottom == LevelZonalBottom.BOTTOM.value:

                plot_Nearside_lat_lon(config, da=da, var_name=var_name)       
                plt.close()
                
                plot_Atlantic_lat_lon(config, da=da, var_name=var_name)
                plt.close()
                # no hovmollers for bottom mas

            elif config.level_zonal_bottom == LevelZonalBottom.ZONAL.value:
                plot_Atlantic_zonal(config, da=da, var_name=var_name)
                plt.close()

                #Hovmollers for zonal data

                plot_hovm_zonal_250_1000m(config, da=da, var_name=var_name)
                plt.close()
                plot_hovm_zonal_2000_3000m(config, da=da, var_name=var_name)
                plt.close()

                

            
        ### Plots for multiple variables...
        if config.numberOfVar == 1:
            return
        

        def plot_lead_lag_40_60N(config:MSSAConfig, da_var1, da_var2, ax, var_name):
            # calculate timeseries of weighted average over area 40N to 60N
            bool1 = ((da_var1.lat > 40) & ( da_var1.lat < 60))
            slice_var1 = da_var1.where(bool1)
            slice_var2 = da_var2.where(bool1)
            if config.level_zonal_bottom == LevelZonalBottom.LEVEL.value:
                ts_var1 = horiz_weighted_averaging(slice_var1, ds_masks.Volume.isel(level = config.level_index))
                ts_var2 = horiz_weighted_averaging(slice_var2, ds_masks.Volume.isel(level = config.level_index))
            elif config.level_zonal_bottom == LevelZonalBottom.VOL.value:
                ts_var1 = horiz_weighted_averaging(slice_var1, ds_masks.Volume.isel(level = -1))
                ts_var2 = horiz_weighted_averaging(slice_var2, ds_masks.Volume.isel(level = -1))

            
            # plot lead-lag 40N-60N surface ------------------------------------------------------------------

            window = int(T[EEOFs_pair_index[0]]/config.red_time)
                        
            lags, correlations = lead_lag_cor(ts_var1, ts_var2,windowsize = window, dt = config.red_time)
            ax.plot(lags, correlations, label=var_name)

            ax.axvline(x=0,color='k',ls='dashed')
            ax.axhline(y=0,color='k',ls='dashed')

            ax.set_xlabel('Lag (years)')
            ax.set_ylabel("Correlation coefficient")
            ax.set_ylim([-1, 1])
            ax.set_xlim(-np.around(T[EEOFs_pair_index[0]]), np.around(T[EEOFs_pair_index[0]]))

            ax.xaxis.set_major_locator(mticker.MultipleLocator(np.around(T[EEOFs_pair_index[0]]) / 4))

            ax.grid()
            ax.set_title("lead - lag cor. surface 40N-60N.  - at lag <0  " + variables_list[0] + " leads the other variables")
            #show()
            plt.legend()


        if ((config.level_zonal_bottom == LevelZonalBottom.LEVEL.value) |(config.level_zonal_bottom == LevelZonalBottom.VOL.value)):
            
            fig,ax = plt.subplots()
            for i in np.arange(1,config.numberOfVar,1):
                plot_lead_lag_40_60N(config, da_var1=list_xr_data_arrays[0], da_var2=list_xr_data_arrays[i], ax=ax, var_name = variables_list[i])
            pdf.savefig(fig)
            plt.close()
            
            
            # simply plot the timeseries -----------------------------------------------------------------------------
            fig, ax = plt.subplots()
            colors = {
                variables_list[0] : 'blue',
                variables_list[1] : 'orange',
                variables_list[2] : 'green',
                variables_list[3] : 'black',
                variables_list[4] : 'purple',
                #variables_list[5] : 'pink',
            }

            for i in range(config.numberOfVar):
                bool40_60 = ((list_xr_data_arrays[i].lat > 40) & ( list_xr_data_arrays[i].lat < 60))
                bool_GIN  = ((list_xr_data_arrays[i].lat > 65 ) & ( list_xr_data_arrays[i].lat < 77 ) & (list_xr_data_arrays[i].lon > 0 ) & ( list_xr_data_arrays[i].lon < 20 ))
                bool_Lab= ((list_xr_data_arrays[i].lat > 53 ) & ( list_xr_data_arrays[i].lat < 65 ) & (list_xr_data_arrays[i].lon > -60 ) & (list_xr_data_arrays[i].lon < -30 ))
            if config.level_zonal_bottom == LevelZonalBottom.LEVEL.value:

                ts_40_60 = horiz_weighted_averaging(list_xr_data_arrays[i].where(bool40_60), ds_masks.Volume.isel(level = config.level_index))
                ts_gin1 = horiz_weighted_averaging(list_xr_data_arrays[i].where(bool_GIN), ds_masks.Volume.isel(level = config.level_index))
                ts_lab1 = horiz_weighted_averaging(list_xr_data_arrays[i].where(bool_Lab), ds_masks.Volume.isel(level = config.level_index))
            elif config.level_zonal_bottom == LevelZonalBottom.VOL.value:
                ts_40_60 = horiz_weighted_averaging(list_xr_data_arrays[i].where(bool40_60), ds_masks.Volume.isel(level = -1))
                ts_gin1 = horiz_weighted_averaging(list_xr_data_arrays[i].where(bool_GIN), ds_masks.Volume.isel(level = -1))
                ts_lab1 = horiz_weighted_averaging(list_xr_data_arrays[i].where(bool_Lab), ds_masks.Volume.isel(level = -1))
                
                ax.plot(ts_40_60.time,ts_40_60, c=colors[variables_list[i]], ls='solid' ,label = variables_list[i] + "40-60N")
                ax.plot(ts_gin1.time,ts_gin1, c=colors[variables_list[i]], ls='dashed' ,label = variables_list[i] + "GIN")
                ax.plot(ts_lab1.time,ts_lab1, c=colors[variables_list[i]], ls='dotted' ,label = variables_list[i] + "Lab")

                plt.xlim(start_time_Hovm, end_time_Hovm)
                plt.legend()
            pdf.savefig(fig)
            plt.close()


        elif config.level_zonal_bottom == LevelZonalBottom.ZONAL.value:
            fig, ax = plt.subplots(2, layout= 'tight')
            for i in np.arange(1,config.numberOfVar,1):
                lead_lag_40_60N_250_1000m(config, da_var1=list_xr_data_arrays[0], da_var2=list_xr_data_arrays[i], ax1=ax[0], ax2=ax[1], var_name = variables_list[i])
            
            pdf.savefig(fig)
            plt.close()

