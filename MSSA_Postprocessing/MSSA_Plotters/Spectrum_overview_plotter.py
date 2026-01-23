from pylab import *
import numpy as np
import netCDF4 as netcdf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors



from mssakit.MSSA_config import MSSAConfig

import netCDF4 as netcdf

def get_sign_freq_pairs(config:MSSAConfig, freq_plus_min_range = 0.0001, power_range_pct = 0.05,significance_pct = 99 , retrieve_freqs = False):

    '''
    takes:
         MSSA config

         retrieve_freqs, choose whether to also return a list of the average frequencies of the significant pairs
    returns:
        list_freq_indx: list with significant EEOF pairs that have frequency in common
        list_freq_power_indx: list with significant EEOf pairs that have freq AND power in common
        
        list_freq_indx_freqs:list with average frequencies of significant EEOF pairs that have freq in common
        list_freq_power_indx_freqs:list with average frequencies of significant EEOF pairs that have freq AND power in common'''


    list_freq_indx = []
    list_freq_power_indx = []
    list_freq_indx_freqs = []

    list_freq_power_indx_freqs = []

    Spectrum_data_path = config.output_directory+'MSSA_Spectra/'+config.get_filename()+'.nc'
    Spectra_data = netcdf.Dataset(Spectrum_data_path, mode='r')

    freq         = Spectra_data.variables['freq'][:]   / config.red_time #adjust for time reduction.
    data_power 		= Spectra_data.variables['data_power'][:]
    

    if significance_pct == 99:
        sign_modes = Spectra_data.variables['isign99'][:].data    
    elif significance_pct == 97.5:
        sign_modes = Spectra_data.variables['isign97_5'][:].data
    elif significance_pct == 95:
        sign_modes = Spectra_data.variables['isign95'][:].data
    elif significance_pct == 90:
        sign_modes = Spectra_data.variables['isign90'][:].data
    Spectra_data.close()
    
    #print(freq)
    #print(sign_modes)
    for i in range(len(sign_modes)):
        #print(i)
        index_upper = sign_modes[i]
        freq_upper = freq[index_upper]
        power_upper = data_power[index_upper]

        for j in np.arange(1,6,1):
            index_new = sign_modes[i+j]
            if index_new >49:
                break
            freq_new = freq[index_new]
            power_new = data_power[index_new]

            #first frequency dependency
            if ((freq_new < freq_upper + freq_plus_min_range) &( freq_new > freq_upper - freq_plus_min_range)):
                #print(f"EEOFs are {index_upper +1} and {index_new +1}")
                list_freq_indx.append([index_upper, index_new])
                list_freq_indx_freqs.append(np.average([freq_upper, freq_new]))

                # power dependency
                if ((power_new < power_upper + power_upper * power_range_pct) & (power_new  > power_upper - power_upper* power_range_pct)):
                    #print(f"EEOFs are {index_upper +1} and {index_new +1} also the same power!")
                    list_freq_power_indx.append([index_upper, index_new])
                    list_freq_power_indx_freqs.append(np.average([freq_upper, freq_new]))

        if sign_modes[i] >20:
            break

    if retrieve_freqs == False:
        return list_freq_indx, list_freq_power_indx
    else:
        return list_freq_indx, list_freq_power_indx, list_freq_indx_freqs, list_freq_power_indx_freqs

def one_over(x):
    """Vectorized 1/x, treating x==0 manually"""
    x = np.array(x, float)
    near_zero = np.isclose(x, 0)
    x[near_zero] = np.inf
    x[~near_zero] = 1 / x[~near_zero]
    return x


def Spectrum_Overview_plot(configs_list,freq_plus_min_range = 0.0001,power_range_pct = 0.05, significance_pct=99):
    cmap = 'tab20c'
    #cmap = 'plasma'
    total_y = len(configs_list)
    y_dy = total_y/15
    mc_norm = mcolors.Normalize(vmin=0, vmax=20)

    fig, axes = plt.subplots(1, figsize=(8,6),layout='constrained')
    list_y_ticklabels = []
    for i, config in enumerate(configs_list):
        list_y_ticklabels.append(config.run_name)

        y_val = i+1

        #config= config_3k_trans_SSS
        list_sign_freq, list_sign_freq_power = get_sign_freq_pairs(config, 
                                   freq_plus_min_range=freq_plus_min_range, power_range_pct=power_range_pct ,
                                   significance_pct=significance_pct)



        Spectrum_data_path = config.output_directory+'MSSA_Spectra/'+config.get_filename()+'.nc'
        Spectra_data = netcdf.Dataset(Spectrum_data_path, mode='r')
        freq         = Spectra_data.variables['freq'][:]   / config.red_time #adjust for time reduction.
        Spectra_data.close()

        EEOF_path = config.output_directory+'EEOFs/'+config.get_filename()+'.nc'
        EEOF_data = netcdf.Dataset(EEOF_path, mode='r')  
        var_expl_mssa = EEOF_data.variables['var_expl_mssa'][:]
        EEOF_data.close()
        
        print(list_sign_freq)
        print(list_sign_freq_power)

        for i in range(len(list_sign_freq)):
            #var_expl = EEOFs.var_expl_mssa[list_sign[i]].sum()
            #print(var_expl.data)
            var_expl = var_expl_mssa[list_sign_freq[i]].sum()
            size = 80

            f = freq[list_sign_freq[i][0]] 
            f2 = freq[list_sign_freq[i][1]]

            if list_sign_freq[i] in list_sign_freq_power:               #Checks if sign pair of equal freq has equal power too
                                                                                                                # if yes, filled markers 
                axes.scatter(f , y=y_val, c=var_expl, s = 40, marker = 'o', norm = mc_norm, cmap=cmap)
                axes.scatter(f2 , y=y_val, c= var_expl, s = 40, marker='o', norm = mc_norm, cmap=cmap)
            else:                                                                                               # else, markers without facecolor (weird code but it works) 
                a =axes.scatter(f , y=y_val, c= var_expl, s = 40, marker = 'o',lw=2, norm = mc_norm, cmap=cmap) 
                b =axes.scatter(f2 , y=y_val, c= var_expl, s = 40, marker='o',lw=2, norm = mc_norm, cmap=cmap)
                a.set_facecolor('none')
                b.set_facecolor('none')
                a.set_edgecolors(a.to_rgba(var_expl))
                b.set_edgecolors(b.to_rgba(var_expl))

            axes.annotate( f"{list_sign_freq[i][0]+1}" , [f,y_val + 0.5* y_dy]) # annotate EEOF number above/below the marker 
            axes.annotate( f"{list_sign_freq[i][1]+1}", [f2,y_val - y_dy])




        '''# plot a line that contains the approximate range of the relevant period based on the window size M  :::  M to M/5
        window_size_real = config.window_size*config.red_time  #change input M (window_size) to true M in years 

        x = [(1/window_size_real) , (1/(window_size_real/5))]  # convert period of M window to frequency window
        axes.fill_between(y1 = 0, y2=total_y +1,x=x, color = 'green', alpha = 0.1)'''





    axes.set_yticks(np.arange(1,len(configs_list)+1,1), list_y_ticklabels)

    axes.set_xlim([0.0001,0.011])
    axes.set_ylim(0,total_y +1)
    axes.set_xlabel('Frequency (years$^{-1}$)')



    twinaxes0 = axes.twiny()
    new_tick_locations = np.array([500,300,200,100,50,10])
    twinaxes0.set_xlabel('Period (years)')

    twinaxes0.set_xticks(one_over(new_tick_locations))
    twinaxes0.set_xticklabels(new_tick_locations)
    twinaxes0.set_xlim(axes.get_xlim())
    twinaxes0.grid(True)

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap), norm=mc_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes, label='Variance Explained (%)')
    cbar.set_ticks(np.arange(0,21,4))

    import matplotlib.lines as mlines

    # Assume you already have your plot and axes
    # ax = plt.gca()  # or use your existing axes

    # Create proxy artists for the legend
    empty_marker = mlines.Line2D([], [], color='none', marker='o',
                                markeredgecolor='black', markerfacecolor='none',
                                markersize=10, lw=2,label='equal freq')
    filled_marker = mlines.Line2D([], [], color='none', marker='o',
                                markeredgecolor='black', markerfacecolor='black',
                                markersize=10, lw=2,label='equal power and freq')

    # Add the legend to your plot
    axes.legend(handles=[empty_marker, filled_marker],
            loc='upper right')#, title='Marker Style')
    

import matplotlib.lines as mlines

def Spectrum_Overview_plot_proxy(configs_list,freq_plus_min_range = 0.0001,power_range_pct = 0.05, significance_pct=99):
    cmap = 'tab20c'
    #cmap = 'plasma'
    total_y = len(configs_list)
    y_dy = total_y/15
    mc_norm = mcolors.Normalize(vmin=0, vmax=20)

    fig, axes = plt.subplots(1, figsize=(8,6),layout='constrained')
    list_y_ticklabels = []
    for i, config in enumerate(configs_list):
        list_y_ticklabels.append(i+1)

        y_val = i+1

        #config= config_3k_trans_SSS
        list_sign_freq, list_sign_freq_power = get_sign_freq_pairs(config, 
                                   freq_plus_min_range=freq_plus_min_range, power_range_pct=power_range_pct ,
                                   significance_pct=significance_pct)



        Spectrum_data_path = config.output_directory+'MSSA_Spectra/'+config.get_filename()+'.nc'
        Spectra_data = netcdf.Dataset(Spectrum_data_path, mode='r')
        freq         = Spectra_data.variables['freq'][:]   / config.red_time #adjust for time reduction.
        Spectra_data.close()

        EEOF_path = config.output_directory+'EEOFs/'+config.get_filename()+'.nc'
        EEOF_data = netcdf.Dataset(EEOF_path, mode='r')  
        var_expl_mssa = EEOF_data.variables['var_expl_mssa'][:]
        EEOF_data.close()
        
        #print(list_sign_freq)
        #print(list_sign_freq_power)

        for i in range(len(list_sign_freq)):
            #var_expl = EEOFs.var_expl_mssa[list_sign[i]].sum()
            #print(var_expl.data)
            var_expl = var_expl_mssa[list_sign_freq[i]].sum()
            size = 80

            f = freq[list_sign_freq[i][0]] 
            f2 = freq[list_sign_freq[i][1]]

            if list_sign_freq[i] in list_sign_freq_power:               #Checks if sign pair of equal freq has equal power too
                                                                                                                # if yes, filled markers 
                axes.scatter(f , y=y_val, c=var_expl, s = 40, marker = 'o', norm = mc_norm, cmap=cmap)
                axes.scatter(f2 , y=y_val, c= var_expl, s = 40, marker='o', norm = mc_norm, cmap=cmap)
            else:                                                                                               # else, markers without facecolor (weird code but it works) 
                a =axes.scatter(f , y=y_val, c= var_expl, s = 40, marker = 'o',lw=2, norm = mc_norm, cmap=cmap) 
                b =axes.scatter(f2 , y=y_val, c= var_expl, s = 40, marker='o',lw=2, norm = mc_norm, cmap=cmap)
                a.set_facecolor('none')
                b.set_facecolor('none')
                a.set_edgecolors(a.to_rgba(var_expl))
                b.set_edgecolors(b.to_rgba(var_expl))

            '''axes.annotate( f"{list_sign_freq[i][0]+1}" , [f,y_val + 0.5* y_dy]) # annotate EEOF number above/below the marker 
            axes.annotate( f"{list_sign_freq[i][1]+1}", [f2,y_val - y_dy])'''




        '''# plot a line that contains the approximate range of the relevant period based on the window size M  :::  M to M/5
        window_size_real = config.window_size*config.red_time  #change input M (window_size) to true M in years 

        x = [(1/window_size_real) , (1/(window_size_real/5))]  # convert period of M window to frequency window
        axes.fill_between(y1 = 0, y2=total_y +1,x=x, color = 'green', alpha = 0.1)'''





    axes.set_yticks(np.arange(1,len(configs_list)+1,1), list_y_ticklabels)

    axes.set_xlim([0.0001,0.011])
    axes.set_ylim(0,total_y +1)
    axes.set_xlabel('Frequency (years$^{-1}$)')



    twinaxes0 = axes.twiny()
    new_tick_locations = np.array([500,300,200,100,50,10])
    twinaxes0.set_xlabel('Period (years)')

    twinaxes0.set_xticks(one_over(new_tick_locations))
    twinaxes0.set_xticklabels(new_tick_locations)
    twinaxes0.set_xlim(axes.get_xlim())
    twinaxes0.grid(True)

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap), norm=mc_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes, label='Variance Explained (%)')
    cbar.set_ticks(np.arange(0,21,4))


    # Assume you already have your plot and axes
    # ax = plt.gca()  # or use your existing axes

    # Create proxy artists for the legend
    empty_marker = mlines.Line2D([], [], color='none', marker='o',
                                markeredgecolor='black', markerfacecolor='none',
                                markersize=10, lw=2,label='equal freq')
    filled_marker = mlines.Line2D([], [], color='none', marker='o',
                                markeredgecolor='black', markerfacecolor='black',
                                markersize=10, lw=2,label='equal power and freq')

    # Add the legend to your plot
    axes.legend(handles=[empty_marker, filled_marker],
            loc='upper right')#, title='Marker Style')