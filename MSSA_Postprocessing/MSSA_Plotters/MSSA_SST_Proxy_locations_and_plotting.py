import numpy as np
import xarray as xr
import netCDF4 as netcdf

from mssakit.MSSA_config import MSSAConfig




def retrieve_MC_configs_and_filenames(input_config: MSSAConfig, grid_pct:int, MC_realizations:int):
    """ 
    gives a list of all the configs and the filenames of each monte carlo of a SST_proxy_location subsampling MSSA run.

    input: 
    input_config: The MC MSSA input_config
    grid_pct, pct of proxy grid cells that were masked
    MC_realizations: number of MC realizations 
    
    returns :
    configs_list, list_filename
    """
    list_filename = []
    configs_list = []
    for i in np.arange(2,MC_realizations,1):
        if i == 1: 
            config_grid_pct = 100
        else:
            config_grid_pct = grid_pct

        run_name = f"{input_config.run_name}_{grid_pct}_pct_MC_n{i}_from_{MC_realizations}"   
        config = MSSAConfig(label=input_config.label, data_path=input_config.data_path, start_time=input_config.start_time,end_time=input_config.end_time,red_time=input_config.red_time, eof_var=input_config.eof_var,window_size=input_config.window_size,realizations=input_config.realizations,sel_region=input_config.sel_region,trend_type=input_config.trend_type,output_directory=input_config.output_directory,norotate=input_config.norotate,
                level_zonal_bottom=input_config.level_zonal_bottom,level_index_range=input_config.level_index_range,
                numberOfVar=input_config.numberOfVar,

                run_name=run_name,
                proxy_cell_pct= config_grid_pct,

                variable=input_config.variable, #, variable2=input_config."temp", nororate_str
                    )
        list_filename.append(config.get_filename())
        configs_list.append(config)
    return configs_list, list_filename
    
def place_in_bins(config:MSSAConfig, list_frequencies_of_interest,list_freq_indx,list_freq_indx_freqs, period_bin_width_fraction = 0.05):
    """
    Places significant EEOF_pairs of a specific run in predetermined bins of interest.
    Works in concert with get_sign_freq_pairs from (from Packages.MSSA_Postprocessing.MSSA_Plotters.Spectrum_overview_plotter)

    Input:
    config: MSSAconfig of a single MC MSSA run.
    list_frequencies_of_interest: predetermined list of frequencies that are used for binning. Based on another MSSA run.

    list_freq_indx: output get_sign_freq_pairs
    list_freq_indx_freqs: output from get_sign_freq_pairs

    period_bin_width_fraction: fraction around which frequency bins are determined (bin determined in period space)


    """


    arr_periods_of_interest = ( 1/ np.array(list_frequencies_of_interest))
    list_lower_bounds_freq_bins = list(1/ (arr_periods_of_interest * (1+period_bin_width_fraction)))
    list_upper_bounds_freq_bins = list(1/ (arr_periods_of_interest * (1-period_bin_width_fraction)))

    dict_periods = {}

    for i in range(len(list_frequencies_of_interest)):
        period = int(arr_periods_of_interest[i])
        #print(period)
        name = f"period_{period}"
        #config_name = f"config_{period}"
        dict_periods[name] = {}
        matching_EEOF = False
        for freqs, EEOFs in zip(list_freq_indx_freqs, list_freq_indx):
            #print(freqs)
            
            if (freqs > list_lower_bounds_freq_bins[i]) & (freqs < list_upper_bounds_freq_bins[i]):
                #print(f"{freqs} inbetween {list_lower_bounds_freq_bins[i]} and {list_upper_bounds_freq_bins[i]} meaning EEOF = {EEOFs}")
                matching_EEOF = True
                matching_EEOF_pair = EEOFs
                
                #print(f"{EEOFs} not in right range for {period}")
        if matching_EEOF == True:
            dict_periods[name]["EEOF_Pair"] = matching_EEOF_pair
        else:
            dict_periods[name]["EEOF_Pair"] = None

    dict_periods['config'] = config
    return dict_periods

from Packages.MSSA_Postprocessing.Retrieve_PC_RC import retrieve_RC
from Packages.MSSA_Postprocessing.MSSA_Plotters.Spectrum_plotter_config_oct import SpectrumPlotter
from IPython.display import clear_output
from Packages.iLC_TOOLS.TB_toolbox import fix_lons_dataset_ilc_to_min180_180
from Packages.iLC_TOOLS.TB_toolbox import horiz_weighted_averaging
from Packages.MSSA_Preprocessing_iLOVECLIM.Loading_Proxies_mask import retrieving_proxy_mask



def calc_horiz_weighted_average(data, mask, volume):
    masked = data.where(mask)
    ts = horiz_weighted_averaging(masked, volume=volume)
    return ts

def calculating_region_averages_dict(dict_1):
    """
    Computes Proxy_relevant regional averages based on an input dictionary.
    Dictionary must have shape:
    dict_1  =  {"MSSA_output1" : {'config': MSSAconfig, "EEOFs_pair" : [5,6], "zscore" :False}
                "MSSA_output2 : {'config': MSSAconfig1, "EEOFs_pair" : [5,6], "zscore" :False}
                etc.}

    Returns the same dictionary with additional RC and timeseries. Keys:
    
    dict_keys(['config', 'EEOFs_pair', 'zscore', 'RC', 'Ice', 'nGIN', 'sGIN', 'GulfMex', 'wAtl', 'eAtl', 'wRidge'])

    useful list for plotting: region_list = ["Ice","nGIN","sGIN","GulfMex","wAtl","eAtl","wRidge"]

    """


    ds_masks = fix_lons_dataset_ilc_to_min180_180(xr.open_dataset('/Users/toonbense/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/iLOVECLIM_code_TB/masks_ilc_tb.nc'))
    Full_Proxy_mask = retrieving_proxy_mask(100, bool180E_180W=True)

    bool_Ice = ((ds_masks.lat > 55) & (ds_masks.lat < 70) & (ds_masks.lon > -33) &  (ds_masks.lon < -13))
    bool_nGIN = ((ds_masks.lat > 70) & (ds_masks.lat < 82) & (ds_masks.lon > 3) &  (ds_masks.lon < 15))
    bool_sGIN = ((ds_masks.lat > 55) & (ds_masks.lat < 70) & (ds_masks.lon > -3) &  (ds_masks.lon < 10))
    bool_GulfMex = ((ds_masks.lat > 15) & (ds_masks.lat < 30) & (ds_masks.lon > -98) &  (ds_masks.lon < -78))
    bool_wAtl = ((ds_masks.lat > 30) & (ds_masks.lat < 50) & (ds_masks.lon > -78) &  (ds_masks.lon < -56))
    bool_eAtl = ((ds_masks.lat > 30) & (ds_masks.lat < 50) & (ds_masks.lon > -20) &  (ds_masks.lon < 2))
    bool_wRidge = ((ds_masks.lat > -30) & (ds_masks.lat < -10) & (ds_masks.lon > 5) &  (ds_masks.lon < 20))

    dict_bool_regions = {"Ice": bool_Ice,
    "nGIN": bool_nGIN,
    "sGIN": bool_sGIN,
    "GulfMex": bool_GulfMex,
    "wAtl": bool_wAtl,
    "eAtl": bool_eAtl,
    "wRidge": bool_wRidge,}

    for name in dict_1:
        if dict_1[name]['EEOFs_pair'] == None:
            print("No significant EEOF")
        else:
            if ((name == "SubsampledGrid") | ((name == "SubsampledGrid_z"))):
                RC_FullGrid = retrieve_RC(config = dict_1[name]['config'], EEOFs_pair=dict_1[name]['EEOFs_pair'], RC_sOR_PCs_or_STDs='RCs', zscore=dict_1[name]['zscore'] )[0]
                RC_FullGrid_Subsampled_zeros = (RC_FullGrid * Full_Proxy_mask)
                dict_1[name]['RC'] =  xr.where(RC_FullGrid_Subsampled_zeros ==0,np.nan,RC_FullGrid_Subsampled_zeros)
            else:
                dict_1[name]['RC'] = retrieve_RC(config = dict_1[name]['config'], EEOFs_pair=dict_1[name]['EEOFs_pair'], RC_sOR_PCs_or_STDs='RCs', zscore=dict_1[name]['zscore'] )[0]

            dict_1[name]['Ice'] = calc_horiz_weighted_average(dict_1[name]["RC"], bool_Ice, ds_masks.Volume.isel(level=-1))
            dict_1[name]['nGIN'] = calc_horiz_weighted_average(dict_1[name]["RC"], bool_nGIN, ds_masks.Volume.isel(level=-1))
            dict_1[name]['sGIN'] = calc_horiz_weighted_average(dict_1[name]["RC"], bool_sGIN, ds_masks.Volume.isel(level=-1))
            dict_1[name]['GulfMex'] = calc_horiz_weighted_average(dict_1[name]["RC"], bool_GulfMex, ds_masks.Volume.isel(level=-1))
            dict_1[name]['wAtl'] = calc_horiz_weighted_average(dict_1[name]["RC"], bool_wAtl, ds_masks.Volume.isel(level=-1))
            dict_1[name]['eAtl'] = calc_horiz_weighted_average(dict_1[name]["RC"], bool_eAtl, ds_masks.Volume.isel(level=-1))
            dict_1[name]['wRidge'] = calc_horiz_weighted_average(dict_1[name]["RC"], bool_wRidge, ds_masks.Volume.isel(level=-1))
    return dict_1


from Packages.iLC_TOOLS.TB_toolbox import lead_lag_cor
import netCDF4 as netcdf


def plot_lead_lag_loc(config:MSSAConfig,EEOF_pair, ts_var1, ts_var2,str1, str2, ax, color):

    Spectrum_data_path = config.output_directory+'MSSA_Spectra/'+config.get_filename()+'.nc'
    Spectra_data = netcdf.Dataset(Spectrum_data_path, mode='r')

    freq  = Spectra_data.variables['freq'][:]   / config.red_time #adjust for time reduction.

    dominant_Period = 1 / freq[EEOF_pair[0]-1]


    window = int(dominant_Period/config.red_time)

    lags, correlations = lead_lag_cor(ts_var1, ts_var2,windowsize = window, dt = config.red_time)

    relative_lags = lags/dominant_Period




    ax.plot(relative_lags, correlations, label=config.variable2, c=color)
    ax.axvline(x=0,color='k',ls='dashed')
    ax.axhline(y=0,color='k',ls='dashed')

    ax.set_xlabel('Lag (phase)')
    ax.set_ylabel("Correlation coefficient")
    ax.set_ylim([-1, 1])
    ax.set_xlim(-1,1)
    ax.set_xticks([-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1], ['-2π', '-1.5π', '-π', '-0.5π', '0', '0.5π', '1π', '1.5π', '2π'])




    ax.grid(True)
    ax.set_title(f"At lag <0  {str1} leads {str2}")

def plot_lead_lag_loc_MC(config:MSSAConfig,EEOF_pair, ts_var1, ts_var2,str1, str2, ax, alpha, color):

    Spectrum_data_path = config.output_directory+'MSSA_Spectra/'+config.get_filename()+'.nc'
    Spectra_data = netcdf.Dataset(Spectrum_data_path, mode='r')

    freq  = Spectra_data.variables['freq'][:]   / config.red_time #adjust for time reduction.

    dominant_Period = 1 / freq[EEOF_pair[1]]


    window = int(dominant_Period/config.red_time)

    lags, correlations = lead_lag_cor(ts_var1, ts_var2,windowsize = window, dt = config.red_time)

    relative_lags = lags/dominant_Period



    ax.plot(relative_lags, correlations, label=config.variable2, alpha=alpha, color = color )
    ax.axvline(x=0,color='k',ls='dashed')
    ax.axhline(y=0,color='k',ls='dashed')

    ax.set_xlabel('Lag (phase)')
    ax.set_ylabel("Correlation coefficient")
    ax.set_ylim([-1, 1])
    ax.set_xlim(-1,1)
    ax.set_xticks([-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1], ['-2π', '-1.5π', '-π', '-0.5π', '0', '0.5π', '1π', '1.5π', '2π'])




    ax.grid(True)
    ax.set_title(f"At lag <0  {str1} leads {str2}")