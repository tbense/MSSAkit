from pylab import *
import numpy as np
import netCDF4 as netcdf
import matplotlib.pyplot as plt



from mssakit.MSSA_config import MSSAConfig
#from mssakit_new.MSSA_config import LevelZonalBottom
from mssakit.MSSA_config import SignTest

def one_over(x):
    """Vectorized 1/x, treating x==0 manually"""
    x = np.array(x, float)
    near_zero = np.isclose(x, 0)
    x[near_zero] = np.inf
    x[~near_zero] = 1 / x[~near_zero]
    return x

def SpectrumPlotter(config: MSSAConfig, significance_pct = 99, axes = None):

    """
        Plots standard mssa spectrum plot and stores it in a pdf for a given significance.
    Input:
    config : MSSAConfig file
    significante_pct = 99 - The significance level for the plot. Choose 90,95,97.5 or 99.

    Output:
        pdf stored in  config.output_directory + "Figures/" + 'PowerSpectrum_' + config.get_string_file_info() + ".pdf"
        
        """
    pdf_filename = config.output_directory + "Figures/" + 'PowerSpectrum_' + config.get_string_file_info() + ".pdf"

    
    Spectrum_data_path = config.output_directory+'MSSA_Spectra/'+config.get_filename()+'.nc'
    Spectra_data = netcdf.Dataset(Spectrum_data_path, mode='r')



    if significance_pct == 99:
        significance_index_upper = 8
        significance_index_lower = 0
    elif significance_pct == 97.5:
        significance_index_upper = 7
        significance_index_lower = 1
    elif significance_pct == 95:
        significance_index_upper = 6
        significance_index_lower = 2
    elif significance_pct == 90:
        significance_index_upper = 5
        significance_index_lower = 3
    else:
        print("put in valid sign_pct 90,95,97.5 or 99.")

    freq         = Spectra_data.variables['freq'][:].data   / config.red_time  #adjust for time reduction.
    data_power 		= Spectra_data.variables['data_power'][:].data
    MC_power	    = Spectra_data.variables['MC_power'][:].data


    if config.signif_test == SignTest.NOISE_BASIS.value:
        data_power = Spectra_data.variables['Noise_base_power'][:].data
        freq = Spectra_data.variables['F_noise_base'][:].data / config.red_time


    if significance_pct == 99:
        sign_modes = Spectra_data.variables['isign99'][:].data
    elif significance_pct == 97.5:
        sign_modes = Spectra_data.variables['isign97_5'][:].data
    elif significance_pct == 95:
        sign_modes = Spectra_data.variables['isign95'][:].data
    elif significance_pct == 90:
        sign_modes = Spectra_data.variables['isign90'][:].data
    sign_modes = sign_modes[sign_modes < 50]

    #print("sign modes at 90%: " + str(Spectra_data.variables['isign90'][:].data))
    #print("sign modes at 95%: " + str(Spectra_data.variables['isign95'][:].data))
    #print("sign modes at 97.5%: " + str(Spectra_data.variables['isign97_5'][:].data))
    print("sign modes at 99%: " + str(Spectra_data.variables['isign99'][:].data))



    
    if axes == None:
        fig, axes = plt.subplots(1, figsize=(8,6), tight_layout=True)



    axes.set_xlim([0.0001,0.02])
    axes.set_xlabel('Frequency (yr$^{-1}$)')
    print(f"freq shape ={freq.shape} and MC_power shape = {MC_power.shape}")
    axes.errorbar(freq, MC_power[4,:],yerr=[MC_power[4,:] - MC_power[significance_index_lower,:],MC_power[significance_index_upper,:] - MC_power[4,:]],
                color = 'r', elinewidth = 1, linewidth = 0.1, ls = '', label = str(abs(significance_pct-100)) + ' - ' + str(significance_pct) + "% CI")

    axes.plot(freq, data_power, 'ks', markerfacecolor = 'none', markeredgecolor='blue', lw=0.5, label = 'T_EOF')
    
    # plot a line that contains the approximate range of the relevant period based on the window size M  :::  M to M/5
    window_size_real = config.window_size*config.red_time  #change input M (window_size) to true M in years 

    #y = [1500,1500]
    #x = [(1/window_size_real) , (1/(window_size_real/5))]  # convert period of M window to frequency window
    #axes.hlines(y=100, xmin= x[0], xmax=x[1], colors='green',alpha=0.5, linestyles='dashed', label='M to M/5')    
    

    axes.set_yscale('log')
    axes.set_ylabel('Power')
    axes.set_title(f'MSSA {config.label} {config.run_name} {significance_pct} {config.signif_test}%')




    if config.signif_test != SignTest.NOISE_BASIS.value:
        axes.scatter(x = freq[sign_modes], y = data_power[sign_modes],
                marker='s', c='blue',label='Significant') #, markerfacecolor = 'blue', markeredgecolor='blue')

        list_freq = list(freq[sign_modes])
        list_power_data = list(data_power[sign_modes])
        list_PC = list(sign_modes+1)
        for X, Y, Z in zip(list_freq[::2], list_power_data[::2], list_PC[::2]):
                # Annotate the points 10 _points_ above and 5 point to the left of the vertex
            axes.annotate('{}'.format(Z), xy=(X,Y), xytext=(-5, 10), ha='center',
                            textcoords='offset pixels')     
        for X, Y, Z in zip(list_freq[1::2], list_power_data[1::2], list_PC[1::2]):
                # Annotate the points 20 _points_ below and 5 point to the left of the vertex
            axes.annotate('{}'.format(Z), xy=(X,Y), xytext=(-5, -20), ha='center',
                            textcoords='offset pixels')
            


    #axes.plot(sign_freq, sign_power, 'sb', lw=0.5, label = 'Significant')
        axes.legend()



    twinaxes0 = axes.twiny()
    new_tick_locations = np.array([500,300,200,100,50,10])
    twinaxes0.set_xlabel('Period (yr)')

    twinaxes0.set_xticks(one_over(new_tick_locations))
    twinaxes0.set_xticklabels(new_tick_locations)
    twinaxes0.set_xlim(axes.get_xlim())

    twinaxes0.grid(True)

    if axes == None:

        plt.show()
        fig.savefig(pdf_filename, dpi=300)
    