from pylab import *
import numpy as np
import netCDF4 as netcdf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines



from mssakit.MSSA_config import MSSAConfig
from TB_toolbox import lead_lag_cor
from MSSA_Postprocessing.Retrieve_PC_RC import retrieve_RC

def get_sign_freq_pairs(config:MSSAConfig, period_range_pct = 0.025, power_range_pct = 0.025,quadrature_range_pct = 0.025,significance_pct = 99, python_index = True, check_quadrature = False,retrieve_freqs = False):

    '''
    takes:
        MSSA config - MSSA_config file
        period_range_pct: range to include in pairs that are equal frequency. E.g., 0.025 selects pairs that both fall within the interval of +- 2.5% of the mean period of the pair.
        power_range_pct: range to include in pairs of equal power. E.g., 0.025 selects pairs that fall within the interval of +- 2.5% of the mean power of the pair.
        quadrature_range_pct: range around quadrature (0.25 phase lag) to be considered in quadrature. E.g., 0.025 selects pairs with a phase lead/lag of [period*(0.25-+0.025)]. 
        However, an additional constrain is in place to ensure the window is at least 2 indexes wide. (e.g., for period = 140, red_time=10. 0.25*140/10 = 3.5 -> ensures [3,4] is considered quadrature even if the calculated range is smaller (e.g., 3.15 - 3.85)) 
        significance_pct: upper confidence interval level of the MSSA-MC sign. test.
        python_index: True: returns lists with python index, False: returns lists with index starting from 1.
        check_quadrature: also checks if eigenpairs are in quadrature
        retrieve_freqs: choose whether to also return a list of the average frequencies of the significant pairs

    returns:
        list_freq_indx: list with significant EEOF pairs that have frequency in common
        list_freq_power_indx: list with significant EEOf pairs that have freq AND power in common
        
        if check_quadrature = Trure, als returns:
        list_freq_quadr_indx list with significant EEOF pairs that have same frequency AND quadrature 
        list_freq_power_quadr_indx:; list with significant EEOF pairs that have frequency AND Power in common AND are quadrature

        elif retrieve_freqs = True (and check_quadrature = False), also returns:
        list_freq_freqs:list with average frequencies of significant EEOF pairs that have freq in common
        list_freq_power_freqs:list with average frequencies of significant EEOF pairs that have freq AND power in common'''


    list_freq_indx = []
    list_freq_power_indx = []
    list_freq_quadr_indx = []
    list_freq_power_quadr_indx =[]

    list_freq_freqs = []
    list_freq_power_freqs = []
    list_freq_quadr_freqs = []
    list_freq_power_quadr_freqs =[]

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

        power_i = data_power[index_upper]

        for j in np.arange(1,6,1): # check the next 5 modes
            if (i+j) == len(sign_modes):
                break
            
            index_new = sign_modes[i+j]
            if index_new >49: # typically only store 50 EOFs.
                break
            freq_new = freq[index_new]
            power_j = data_power[index_new]


            #period_center = 1/freq_center
            period_center = (1/freq_upper +1/freq_new)/2 #average period
            freq_lower_lim = 1 / (period_center * (1+period_range_pct))
            freq_upper_lim = 1 / (period_center * (1-period_range_pct))

            #first frequency dependency. both freq_upper and freq_new must be within the range.
            if ((freq_upper < freq_upper_lim) & (freq_upper > freq_lower_lim) & (freq_new < freq_upper_lim) & (freq_new > freq_lower_lim)):
                list_freq_indx.append([index_upper, index_new])
                list_freq_freqs.append(np.average([freq_upper, freq_new]))
                eq_power = False
                
                
                # power dependency. both powers must be within the range from the average power
                power_center = ( power_i + power_j)/2
                upper_power_lim = power_center * (1 + power_range_pct)
                lower_power_lim = power_center * ( 1 - power_range_pct)
                if ((power_i <upper_power_lim) & (power_i  > lower_power_lim)& (power_j <upper_power_lim) & (power_j  > lower_power_lim)):
                    list_freq_power_indx.append([index_upper, index_new])
                    list_freq_power_freqs.append(np.average([freq_upper, freq_new]))
                    eq_power = True


                if check_quadrature:
                #quadrature dependency
                    PCs =retrieve_RC(config=config, EEOFs_pair=[index_upper+1, index_new+1], RC_sOR_PCs_or_STDs="PCs") #array with the pcs[time,pc_n]
                    #plt.plot(PCs[:,0])
                    #plt.plot(PCs[:,1])
                    #plt.title(f"{index_upper},{index_new}")
                    #plt.show()
                    #print([index_upper+1, index_new+1])
                    period = int(period_center)
                    lags, cors = lead_lag_cor(PCs[:,0], PCs[:,1], dt=1, windowsize = int( (period/config.red_time)/2) )
                    max_lag = np.abs(lags[np.argmax(cors)]) #selects absolute lag associated to max cor_coef
                    min_lag =np.abs(lags[np.argmin(cors)]) # same for min


                    # determine quadrature cutoffs.  
                    quad_lag = 0.25*(period/config.red_time)
                    quad_min = np.floor(quad_lag)
                    quad_max = np.ceil(quad_lag)

                    quad_lower_range = (period/config.red_time) * (0.25-quadrature_range_pct)
                    quad_upper_range = (period/config.red_time) * (0.25+quadrature_range_pct)

                    index_lower_lim =   np.min([quad_min, quad_lower_range])
                    index_upper_lim =   np.max([quad_max, quad_upper_range])

                    if (( (max_lag >=index_lower_lim) & (max_lag<= index_upper_lim)) | ((min_lag >=index_lower_lim) & (min_lag<= index_upper_lim))):
                        list_freq_quadr_indx.append([index_upper, index_new])
                        list_freq_quadr_freqs.append(np.average([freq_upper, freq_new]))
                        #print([index_upper, index_new])
                        if eq_power:
                            list_freq_power_quadr_indx.append([index_upper,index_new])
                            list_freq_power_quadr_freqs.append(np.average([freq_upper, freq_new]))



        if sign_modes[i] >20:
            break

    if python_index != True: # 'index starting from 1 instead of 0'
        new_list_f = (np.array(list_freq_indx)+1).tolist()
        new_list_p = (np.array(list_freq_power_indx)+1).tolist()
        list_freq_indx=new_list_f
        list_freq_power_indx = new_list_p

        if check_quadrature == True:
            new_list_f_q = (np.array(list_freq_quadr_indx)+1).tolist()
            new_list_f_p_q = (np.array(list_freq_power_quadr_indx)+1).tolist()
            list_freq_quadr_indx = new_list_f_q
            list_freq_power_quadr_indx = new_list_f_p_q

    if (retrieve_freqs == False) & (check_quadrature==False): 
        return list_freq_indx, list_freq_power_indx
    
    elif (retrieve_freqs == False) & (check_quadrature == True):

        return list_freq_indx, list_freq_power_indx, list_freq_quadr_indx, list_freq_power_quadr_indx
    
    elif (retrieve_freqs == True) & (check_quadrature == False):
        return list_freq_indx, list_freq_power_indx, list_freq_freqs, list_freq_power_freqs

    elif ((retrieve_freqs == True) & (check_quadrature == True)):
        return list_freq_indx, list_freq_power_indx, list_freq_quadr_indx, list_freq_power_quadr_indx, list_freq_freqs, list_freq_power_freqs, list_freq_quadr_freqs, list_freq_power_quadr_freqs


def one_over(x):
    """Vectorized 1/x, treating x==0 manually"""
    x = np.array(x, float)
    near_zero = np.isclose(x, 0)
    x[near_zero] = np.inf
    x[~near_zero] = 1 / x[~near_zero]
    return x


def Spectrum_Overview_plot(configs_list,period_range_pct = 0.025,power_range_pct = 0.025,quadrature_range_pct = 0.025, significance_pct=99, check_quadrature = True,ax = None):
    """
    Plots spectrum overview figure. 

    Takes:
    configs_list: list of MSSAconfig files
    period_range_pct = 0.025, 
    power_range_pct = 0.025, 
    quadrature_range_pct = 0.025, 
    significance_pct=99, 
    check_quadrature = True, if quadrature check should be included
    ax: ax to plot on

    Returns:
    MSSA Overview figure :)
    """
    
    
    
    cmap = 'tab20c'
    #cmap = 'plasma'
    total_y = len(configs_list)
    y_dy = total_y/15
    mc_norm = mcolors.Normalize(vmin=0, vmax=20)




    # if ax instance is provided, plot on that ax instance. Otherwise make a new figure.
    if ax == None:
        fig, axes = plt.subplots(1, figsize=(8,6),layout='constrained')
    else:
        axes = ax

    list_y_ticklabels = []
    for i, config in enumerate(configs_list):
        list_y_ticklabels.append(config.run_name)

        y_val = i+1

        #config= config_3k_trans_SSS
        if check_quadrature:
            list_sign_freq, list_sign_freq_power, list_sign_freq_quad, list_sign_freq_power_quad = get_sign_freq_pairs(config, 
                                   period_range_pct=period_range_pct, power_range_pct=power_range_pct ,
                                   significance_pct=significance_pct, quadrature_range_pct=quadrature_range_pct,check_quadrature=check_quadrature)
        else:
            list_sign_freq, list_sign_freq_power = get_sign_freq_pairs(config, 
                                   period_range_pct=period_range_pct, power_range_pct=power_range_pct ,significance_pct=significance_pct)


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

            if check_quadrature: #consider freq, quad, and power.
                if list_sign_freq[i] in list_sign_freq_power_quad:               #Checks if sign pair of equal freq has equal power AND quadrature.                                         
                    axes.scatter(f , y=y_val, c=var_expl, s = 50, marker = '*', norm = mc_norm, cmap=cmap)
                    axes.scatter(f2 , y=y_val, c= var_expl, s = 50, marker='*', norm = mc_norm, cmap=cmap)
                elif list_sign_freq[i] in list_sign_freq_power:
                    axes.scatter(f , y=y_val, c=var_expl, s = 40, marker = '^', norm = mc_norm, cmap=cmap)
                    axes.scatter(f2 , y=y_val, c= var_expl, s = 40, marker='^', norm = mc_norm, cmap=cmap)   
                elif list_sign_freq[i] in list_sign_freq_quad:
                    axes.scatter(f , y=y_val, c=var_expl, s = 40, marker = 's', norm = mc_norm, cmap=cmap)
                    axes.scatter(f2 , y=y_val, c= var_expl, s = 40, marker='s', norm = mc_norm, cmap=cmap)            
                elif list_sign_freq[i] in list_sign_freq:                                                                                              
                    a =axes.scatter(f , y=y_val, c= var_expl, s = 40, marker = 'o',lw=2, norm = mc_norm, cmap=cmap) 
                    b =axes.scatter(f2 , y=y_val, c= var_expl, s = 40, marker='o',lw=2, norm = mc_norm, cmap=cmap)

            else: #only consider freq and power
                if list_sign_freq[i] in list_sign_freq_power:
                    axes.scatter(f , y=y_val, c=var_expl, s = 40, marker = '^', norm = mc_norm, cmap=cmap)
                    axes.scatter(f2 , y=y_val, c= var_expl, s = 40, marker='^', norm = mc_norm, cmap=cmap) 
                elif list_sign_freq[i] in list_sign_freq:                                                                                              
                    a =axes.scatter(f , y=y_val, c= var_expl, s = 40, marker = 'o',lw=2, norm = mc_norm, cmap=cmap) 
                    b =axes.scatter(f2 , y=y_val, c= var_expl, s = 40, marker='o',lw=2, norm = mc_norm, cmap=cmap)



            axes.annotate( f"{list_sign_freq[i][0]+1}" , [f,y_val + 0.5* y_dy]) # annotate EEOF number above/below the marker 
            axes.annotate( f"{list_sign_freq[i][1]+1}", [f2,y_val - y_dy])


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
    # Create proxy artists for the legend
    #empty_marker = mlines.Line2D([], [], color='none', marker='o',
    #                            markeredgecolor='black', markerfacecolor='none',
    #                            markersize=10, lw=2,label='equal freq')
    filled_marker_f_p_q = mlines.Line2D([], [], color='none', marker='*',
                                markeredgecolor='black', markerfacecolor='black',
                                markersize=10, lw=2,label='Freq, power, quad.')
    filled_marker_f_q = mlines.Line2D([], [], color='none', marker='s',
                                markeredgecolor='black', markerfacecolor='black',
                                markersize=10, lw=2,label='Freq., quad.')
    filled_marker_f_p = mlines.Line2D([], [], color='none', marker='^',
                                markeredgecolor='black', markerfacecolor='black',
                                markersize=10, lw=2,label='Freq., power')

    filled_marker_f = mlines.Line2D([], [], color='none', marker='o',
                                markeredgecolor='black', markerfacecolor='black',
                                markersize=10, lw=2,label='Freq.')
    # Add the legend to your plot
    #axes.legend(handles=[empty_marker, filled_marker],
    if check_quadrature:
        axes.legend(handles=[filled_marker_f_p_q,filled_marker_f_q, filled_marker_f_p,filled_marker_f],
            loc='upper right')#, title='Marker Style')
    else:
        axes.legend(handles=[filled_marker_f_p,filled_marker_f],
            loc='upper right')#, title='Marker Style')


