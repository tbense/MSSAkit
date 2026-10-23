from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum
#import numpy as np

class LevelZonalBottom(Enum):
        LEVEL = "level"
        ZONAL = "zonal"
        BOTTOM = "bottom"
        Three_D = "3D"
        VOL = "VOL"
        
class SignTest(Enum):
        DATA_BASIS = "data"
        NOISE_BASIS = "noise"
        PROCRUSTES = "Procrustes"

@dataclass
class MSSAConfig:
    """
    TB - Oct 2025.

    Configuration class for mssa runs.
    Applied in all mssa functions!

    ----------------
    Input: \n
        Required

            label: str - simulation name \n
            run_name: str - mssa run name\n
            datafile_path: str - path to datafile used in run. \n
            start_time: int - start time of mssa run. E.g., start_time = 1500 selects data after 1500\n
            end_time: int - end time of mssa run.\n
            eof_var: float - variance explained by pre PCA analysis step. \n
            window_size: int - Window size M. Must be adjusted for potential time reductions. E.g., for a window size of 500 with red_time = 10 -> M = 50 \n
            sel_region: str - region mask selected. Valid options: Atlantic, West_Atlantic, East_Atlantic, Arctic, Atlantic_Arctic,Southern, or Atlantic_Arctic_Southern, Global_Ocean\n
            level_zonal_bottom: - depth selection, indicate : level, zonal, bottom, 3D, VOL\n
            output_directory: str - Path to output directory. Output directory must include 3 directories: Figures, MSSA_Spectra, EEOFs \n
            numberOfVar: int - number of variables used\n
            variable: str - variable name of first variable\n

    --------------
        Optional: \n
            variable2: str \n
            variable3: str \n
            variable4: str \n
            variable5: str\n

            level_index: int  \n
            level_index_range: list\n
            norotate: list - standard empty. Otherwise, put the index of EEOFs (e.g.,[0,1,2,3,4,5,6] etc.) that are not to be included in the next round of red_noise fitting (ie, all significant EEOFs) to test composite null hypothesis.\n
            norotate_str: str = ""          standard empty string. Otherwise, put in words which STPCs are in norotate\n
            realizations: int - standard 100. Amount of Monte Carlo realizations\n
            red_time: int = 10               time reduction factor. Takes average over timeseries, e.g. 10 = decadal average.\n
            trend_type: int = 2              trend polynomial that is removed from each grid cell. standard quadratic detrend\n
            n_rec_comp = 50                  standard number of reconstructed components (ie only stores RCs for first 50 EEOFs)\n
            pre_pca_bool = True              indicate if pre pca required\n
            varimax_bool = True              indicate if varimax should be applied\n

    """
    
    #standard instances
    label: str
    run_name: str
    data_path: str
    start_time: int
    end_time: int
    eof_var: float
    window_size: int
    sel_region: str
    level_zonal_bottom: LevelZonalBottom  # Only specific values allowed

    output_directory: str 

    signif_test : SignTest

    numberOfVar: int
    variable: str
    norotate: Optional[List]        # standard empty list

    ## optional instances after standard instances
    variable2: Optional[str] = None
    variable3: Optional[str] = None
    variable4: Optional[str] = None
    variable5: Optional[str] = None

    level_index: Optional[int] = None  # level_index corresponding to depth level, e.g. for ILV surface = -1
    level_index_range: Optional[list] = None # list with the index of the level slice, e.g. [5,10] 

    rho_filename: Optional[str] = None
    mld_filename: Optional[str] = None
    pre_pca_bool: Optional[bool] = field(default=True)           #
    varimax_bool: Optional[bool] = field(default=True)      

    ## instances with standard value last!

    norotate_str: str = ""          #standard empty string
    realizations: int = 100

    period_str: str = "years"       #standard years for ilc
    red_time: int = 10              #standard 10 for centennial variability
    trend_type: int = 2             #standard quadratic detrend
    #season_remove : int = 0         #standard (0=False), not necessary for annual data
    n_rec_comp = 50                 #standard number of reconstructed components (ie only stores RCs for first 50 EEOFs)

    proxy_cell_pct: int = 100   #standard 100 %

    
    #___________________________________________________________________________

    # retrieve string for selected variables
    def get_string_variables(self) -> str:
        """
        Retrieves string of all variables used in the mssa run
        """
        if self.numberOfVar == 1:
            return self.variable
        elif self.numberOfVar == 2:
            return f"{self.variable}_and_{self.variable2}"
        elif self.numberOfVar == 3:
            return f"{self.variable}_and_{self.variable2}_and_{self.variable3}"
        elif self.numberOfVar == 4:
            return f"{self.variable}_and_{self.variable2}_and_{self.variable3}_and_{self.variable4}"
        elif self.numberOfVar == 5:
            return f"{self.variable}_and_{self.variable2}_and_{self.variable3}_and_{self.variable4}_and_{self.variable5}"
        else:
            raise ValueError("numberOfVar must be 1,2,3,4 or 5")
    #retrieve string for selection

    def get_string_depth_zonal_bottom(self) -> str:
        """
        retrieves string that represents the depth level or range used in the mssa run
        """
        if self.level_zonal_bottom == LevelZonalBottom.LEVEL.value:
            if self.level_index == -1:
                return "surface"
            else:
                return f"level_{self.level_index}"
            
        elif self.level_zonal_bottom == LevelZonalBottom.ZONAL.value:
            return "latdepth"
        
        elif self.level_zonal_bottom == LevelZonalBottom.BOTTOM.value:
            return "bottom_cells"
        elif self.level_zonal_bottom == LevelZonalBottom.VOL.value:
            return f"vol_ave_i{self.level_index_range[0]}_to_i{self.level_index_range[1]}"
        elif self.level_zonal_bottom == LevelZonalBottom.Three_D.value:
            return "3D_field"
        else:
            raise ValueError("Invalid level_zonal_bottom value")

    def get_string_file_info(self) -> str:
        '''
        retrieves a string with the file information that represents the settings of the mssa run.

        '''
        variables = self.get_string_variables()
        level_string = self.get_string_depth_zonal_bottom()

        return (
            f"{self.run_name}_{self.label}_{variables}_"
            f"{self.start_time}_{self.end_time}_"
            f"{self.red_time}_"
            f"{self.trend_type}_{self.eof_var}_"
            f"{self.window_size}_{self.realizations}_"
            f"{level_string}_{self.norotate_str}_"
            f"{self.signif_test}_{self.varimax_bool}"
        )
    
    def get_filename(self) -> str:
        ''' retrieves the filename associated with the mssa run. 

        same as f"MSSA_{self.get_string_file_info}"
        '''

        return ("MSSA_" + self.get_string_file_info())
    
    def get_list_variables(self):
        '''
        retrieves a list with strings of the variables used in the mssa run.
        '''
        var_list = ([self.variable, self.variable2, self.variable3, self.variable4, self.variable5])
        updated_var_list =[]
        for var_name  in(var_list):
            if var_name != None:
                updated_var_list.append(var_name)
        return updated_var_list