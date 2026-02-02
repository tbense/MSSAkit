import numpy as np
from tqdm import tqdm
import glob, os
import netCDF4 as netcdf
from scipy.signal import lfilter
from scipy.linalg import hankel
from numpy.linalg import svd

from mssakit.mssakit_tools.pca import pca
import mssakit.mssakit_tools.toolbox as tool
import mssakit.mssakit_tools.time_analysis as time_a

from mssakit.MSSA_config import MSSAConfig
from mssakit.MSSA_config import LevelZonalBottom
from mssakit.MSSA_config import SignTest



class mssa_new:
    '''
    Created by TBense. Last updated - 23-01-2026.

    Performs MSSA decomposition on multivariate time series.
    Strongly based on https://github.com/anitasaraswati/MSSAkit  Saraswati, A. T., & de Viron, O. (2023). anitasaraswati/MSSAkit: MSSAkit v1.0.0 (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.10377708
    
    ~ Adjustments (Jan/2026) TB: 
        ~ integration with MSSAConfig, including automatic storing of Spectrum and EEOF data in NC files.
        ~ integration for running multiple variables within one MSSA run
        ~ integration for selecting a number of PCs in prepca based on explainec variance %
        ~ Changed way of storing confidence interval information.
        ~ addition of Allen and Smith 1996 and Allen and Robertson 1996 significance test (data base, noise base). 
        ~ Change in final step Procrustes rotation:   #Wn_wrong[n,:] = np.diag(Te.T@np.diag(wn)@Te)**2  should be    Wn[n,:] = np.diag(Te.T@np.diag(wn**2)@Te) 
                                                      # (see eq. 11/15 GG15, Lambda_r^_{PSigma} = T' Lambda R T    - where Lambda R = eigenvalues = singular_values squared = wn**2. Squaring (T' wn T) affects outcome.)
        ~ Change in estimation of dominant frequency from PCs to vectors from eigenbase.


    ------------------------------------------------------------
                                              
    Multivariate time series should have observations in columns and time series
    indices in rows !!! -> array.shape (time, channels)

    ------------------------------------------------------------
    Parameters:

        config: MSSAConfig file created using mssakit.MSSA_config import MSSAConfig 
        config file contains most information about the mssa_run, including: 
        eof_var, window_size, variables, information about the data, realizations, red_time, norotate. \n

        ts : list of 2D np.arrays. Each array represents a variable, where the array has shape (time, channels)
            Time series array.

        dims: list of dimensions [dim_i, dim_j]. Here dim_i and j represent np.arrays of floats with the grid_cell values, e.g, lon and lat. for 3D mssa, dim_k contains depth levels. 

        masked_field: array that contains the masked grid_cell data_field, necessary for reconstructing the RCs into an iloveclim GRID.          

        t: time array associated with the input_data times (ts)

    ------------------------------------------------------------
    Attributes:
    
        ts : list of 2D np.arrays. Each array represents a variable, where the array has shape (time, channels)
            Time series array.
        t: time np.array associated with the input_data times (ts)

        dims: list of dimensions [dim_i, dim_j]. Here dim_i and j represent np.arrays of floats with the grid_cell values, e.g, lon and lat. for 3D mssa, dim_k contains depth levels. 

        masked_field: array that contains the masked grid_cell data_field, necessary for reconstructing the RCs into an iloveclim GRID.   
        
        n_pc_list: list of the number of PCs that are stored in the prePCA phase for each variable.

        ts_pc : numpy.ndarray
            The set of PCs that is selected from the prePCA before the MSSA, if pre_pca is selected. Stacks PCs of all variables on top of each other
        M : int
                The given windows size used in the MSSA.    
        L : L = N - M + 1,represents amount of timesteps when the lagged window is considered
            
        trajectory_matrix : numpy.ndarray
                The resulting trajectory matrix in the MSSA
        left_vectors : numpy.ndarray
            The resulting left singular vectors (EU) from SVD of the trajectory matrix, also familiar with Time-EOF.
    
        right_vectors : numpy.ndarray
            The resulting right singular vectors (EV) from SVD of the trajectory matrix, also familiar with SpaceTime-EOF.
            
        singularvalue : numpy.array
            The resulting singular values (EW) from SVD of the trajectory matrix.
        
        eigenvalue : numpy.array
            The eigenvalues of the trajectory matrix (singularvalue ** 2).

        var_expl_mssa : numpy.array
            The variance explained by each ST-PC: eigenvalue / eigenvalue.sum() * 100

        PC : numpy.ndarray
            The Time-PC from the MSSA.
        
        RC : numpy.ndarray
            The reconstruction of MSSA in the original multi-channel data
            (or PCA components if the pre_pca is selected). The dimension of this 
            matrix is (N, D, rank), where N is the number of observations, D is 
            the number of channels, and rank is the number of components selected to keep.
        
        RC_spatial : list of np.ndarrays corresponding to the RCs of each input variable.
            The arrays contain the reconstruction of MSSA in the original multi-channel data, if the pre_pca is selected.
            The dimension of this matrix is (N, D, rank), where N is the number
            of observations (time), D is the number of channels, and rank is the
            number of components selected to keep.
        
        F : numpy.array
            The dominant frequency of the T-EOF.
        
        T : numpy.array
            The dominant period of the T-EOF.
        
        sigma_T : numpy.array
            The period uncertainty of the obtained dominant period.        
        
    '''


    def __init__(self,
                 config: MSSAConfig,
                 ts,
                 dims,
                 masked_field,          
                 t,

                 sampling_interval=1): 
        
        self.masked_field = masked_field
        self.dims = dims
        self.time = t
        self.ts = ts
        
        
        # For MSSA, usually the first step is to perform a normal PCA on the post-processed data to reduce the channel size from. x EOFs are chosen as input to MSSA.

        def pre_pca_funct(ts, eof_var):
            """
            Performs pca.

            Input:
            ts: an an array of shape (time, channels).
            eof_var : amount of variance the PCs must explain.

            Returns:
            y - selected PCs
            vr - right singular vectors from PCA
            n_pc - number of PCs that explain eof_var of the variance.

            """
            #print(ts.shape)
            #print(ts)
            ts_pca = pca(ts)
            pc = ts_pca.PC
            
            var_expl = ts_pca.variance_captured_perc.cumsum()*100

            if eof_var == 100: #select all pcs as input (y)
                y = pc
                n_pc = pc.shape[1]
                vr = ts_pca.v

            elif eof_var < 100:
                idxx=0
                #print(f"var expl has length: {len(var_expl)}")
                for i in range(len(var_expl)):
                    if ((var_expl[i] > eof_var)):
                        if (i > 4):
                            print(f"hello 1 {i} and {eof_var}")
                            idxx = i
                            break
                n_pc = idxx+1
                y = pc[:,:n_pc]         # selected amount of PCs
                                        # PCs are scaled by singular values (!) to ensure variance retained
                vr = ts_pca.v[:,:n_pc]  # right singular vectors from PCA

            print("Selected " + str(n_pc) + " PCs that together explain " +str( np.around(var_expl[n_pc-1],3 )) + " % of the variance")
            return y, vr, n_pc

        y_raw = []
        vr = []
        n_pc_list = []
        for i in range(config.numberOfVar): # Perform a pre_pca that returns PCs, right singular vectors, and n_pc for each variable corresponding to the PCs that explain eof_var %.
            if config.pre_pca_bool: 
                y_i, vr_i, n_pc = pre_pca_funct(ts[i], eof_var=config.eof_var)            
                y_raw.append(y_i)
                vr.append(vr_i)
                n_pc_list.append(n_pc)
            else:
                y_raw.append(np.copy(ts[i]))
        
        self.n_pc_list = n_pc_list

        y_list = []
        D=0


        for i in range(config.numberOfVar):  # remove mean from each y (each PC)  
            N_i, D_i = y_raw[i].shape
            y_demean = np.copy(y_raw[i])
            for k in range(D_i):  
                y_demean[:,k] -= np.mean(y_raw[i][:,k])
            y_list.append(y_demean)


            D = D + D_i # amount of input channels increases with variables

            N=N_i  # N is independent of numberOfVar
        
        
        y = np.concatenate(y_list, axis=1) # merges all pcs together to create mssa input array ts_pc of shape (time, selected_pre_pcas)
        self.ts_pc = np.copy(y)


        print(y.shape)
        print('----------')
        print(y)

        M = int(config.window_size)
        self.M = M
        
        L = int(N-M+1)   # L = N' = N - M +1 # represents amount of timesteps when the lagged window is considered
                    # independent of numberOfVar
        self.L = L
        #print(f"L = {L}")
        #print(f"M = {M}")
        #print(f"D = {D}")
        # creating trajectory matrix
        idx = hankel(np.arange(L),np.arange(N-M,N))
        xtde = np.zeros((L,M,D),complex) # trajectory matrix X 
        for d in range(D):
            xin = y[:,d]    
            xtde[:,:,d] = xin[idx]            
        self.trajectory_matrix = xtde

        xtde1 = np.copy(xtde)
        #print(xtde)
        xtde = np.reshape(xtde.swapaxes(1,2),(N-M+1,D*M)) # reshape trajectory matrix into X (L, M*D) , ie by stacking the lagged copies of the matrix behind each other
        

        #  perform MSSA by applying SVD to the trajectory matrix, devided by D*M to make the singular values squared compare directly to variance explained.
        #  Applying SVD directly on trajectory matrix X gives us the singular vectors (left, EU and right EV^T) which represent T-EOFs and S-EOFs as well as the singular values Lamda_D. right singular vectors EV=E_d . (eq 4 AS96), eq2 GG15


        EU,EW,EV = svd(xtde/np.sqrt(D*M),full_matrices=False)
        EV = EV.T
        


        # rank sort from
        ks = EW.argsort()[::-1]
        EW = EW[ks]
        EV = np.real(EV[:,ks])
        EU = np.real(EU[:,ks])
        eigval = EW**2          # eigenvalues = singular values squared.
        
        if config.varimax_bool:            
            # Rotate only the components with % of eigenvalues > 1%
            Sm = np.nonzero(eigval/sum(eigval)>0.01)[0][-1]+1           # select the index Sm where % explained>= 1%
            
            EVsc = EV[:,:Sm]@np.diag(EW[:Sm])                           # cross product eigenvectors EV with singular values to scale them appropriately
            EVr = np.zeros((M,D,Sm))                                    # create empty arrar of size M,D3,Sm and place values of EVsc in there
            for k in range(Sm):
                EVr[:,:,k] = np.reshape(EVsc[:,k],(D,M)).T            # EVr corresponds to S in Groth and Ghil 2011
                
            dummy,Tv = tool.varimax_fun(np.real(EVr))                 # Apply varimax rotation, see toolbox.varimax_fun for annotations. Also Groth & Ghil 2011. 
                                                                   # Tv = Rotation matrix T from eq. 9 in (GG2011)
            EV0 = np.copy(EV)
            EU0 = np.copy(EU)

            EV[:,:Sm] = EV[:,:Sm] @ Tv                              # Update the first Sm eigenvectors (non normalised = EV) by applying formula 9 (GG11) 
            EW[:Sm] = np.diag(Tv.T@np.diag(EW[:Sm])@Tv)                   # Formula 12 (GG11)
            EU[:,:Sm] = EU[:,:Sm]@Tv                                # similar to EV       
    
            # sort again EW
            ks = EW.argsort()[::-1]
            EW = np.real(EW[ks])
            EV = EV[:,ks]
            EU = EU[:,ks]            
            eigval = EW**2
    
            EV = np.real(EV)
            EU = np.real(EU)

            ll = np.arange(0,D*M,M)
            
            Xv = (EU@np.diag(EW)@EV.T)*np.sqrt(D*M)                      #recompute the trajectory matrix by a reverse SVD and reversing the normalisation (Why D*M)
            Xtdev=np.reshape(Xv,(L,D,M)).swapaxes(2,1)                # shape trajectory matrix X to 3D (L, M,D)
            yv=Xv[:,ll]
            yvv=Xv[L-M+1:,ll+M-1]            
            yv=np.concatenate((yv,yvv),axis=0)            
    
            ## not used : ew=np.diag(EW)
            
            self.trajectory_matrix = Xtdev # overwrite trajectory matrix with new one after varimax.
            
        self.left_vectors = EU
        self.right_vectors = EV
        self.singularvalue = EW
        self.eigenvalue = eigval
        
        self.var_expl_mssa =  np.around((self.eigenvalue / self.eigenvalue.sum() * 100),3) # store amount of variance that is explained by each ST-PCs

        # % get V and R in all pc
        NC = config.n_rec_comp
        a=np.zeros((N,D))
        A=np.zeros((L,NC));
        RC=np.zeros((N,D,NC))
        ia=0


        #The below loop implements eq 3 and 4 from GG2011 
        #a = pc = sum over D and M,  e_dk * x_d.  aka lfilter Ej * yr (normalised by 1 = not normalised)
        # RC = sum over D and M (implicit in hstack pc)
        # RC = a_k * e_dk   aka lfilter pc * Ej, normalised by M (see 1/M before the summation in eq4)     

        for o in range(NC) :
            Ej=(EV[:,o].reshape((D,-1))).T
            for j in range(D):
                a[:,j]=lfilter(np.flip(Ej[:,j]),1,y[:,j])
            pc= np.sum(a,axis=1)[M-1:]
            A[:,ia]=pc

            pc=np.hstack((pc,np.zeros(M-1)))
            
            for j in range(D):                   
                RC[:,j,ia]=lfilter(Ej[:,j],M,pc)           #normalisation with factor M (ie * 1/M)    (text under eq. 52 Ghil et al. 2002)
            ia+=1    
        for i in range(0,M-1):                  ## some sort of edge corretion for the sliding window ->  (eq 52 -> 11/12 G2002)
            RC[i,:,:]=RC[i,:,:]*M/(i+1)
            RC[N-i-1,:,:]=RC[N-i-1,:,:]*M/(i+1)
            
            
        self.PC = A   # 2 / 3 in Groth&Ghil 2011
        self.RC = RC  # 4 GG2011      
        


        Rx_list =[]
        n_pc_cumsum = [0]
        for i in range(config.numberOfVar): # for each variable, select the RCs that are associated with that variable. Store in lists.
            n_pc_cumsum.append( n_pc_cumsum[-1] + n_pc_list[i] )
                                
            if config.pre_pca_bool:                          #Convert Reconstructed components from current form (like the pre-pca output) to the input before pre-pca (ie, (time, channels, RCs))
                Rx = np.zeros((ts[i].shape[0],ts[i].shape[1],NC))
                vt = np.transpose(vr[i])
                
                #print("-----------")
             

                for j in range(NC):
                        Rx[:, :,j]= (RC[:,n_pc_cumsum[i]:n_pc_cumsum[i+1],j] @ vt)  #/w*stdx    ,retrieve the parts corresponding to variable 1  up to n_pc
                Rx_list.append(Rx)


        self.RC_spatial = Rx_list



        #        self.F = time_a.dominant_freq(A,fs=sampling_interval) 
        #        self.T, self.sigma_T = time_a.dominant_period(A,fs=sampling_interval,uncertainty=True)

        if M*D >L: #left vectors are eigenbase - > EU
                self.F = time_a.dominant_freq(EU,fs=sampling_interval) 
                self.T, self.sigma_T = time_a.dominant_period(EU,fs=sampling_interval,uncertainty=True)

        elif L > M*D: #right vectors form eigenbase -> EV
                self.F = time_a.dominant_freq(EV,fs=sampling_interval) 
                self.T, self.sigma_T = time_a.dominant_period(EV,fs=sampling_interval,uncertainty=True)        
      
        
    def mcmssa(self, config : MSSAConfig):        
        ''' Applying Monte Carlo to the MSSA. Significance level is stored at 90,95,97.5 and 99 % of the interval between 0 and 100%
        ------------------------------------
        Input parameters stored in self and config file. 

        MSSAconfig includes choices on:
            
            Realizations : int
                Number of surrogates. The default is 100.
            
            norotate : list
                The list of index of non-rotated T-EOF. The default is []
                
        ---------------------------------------------
        Atributes:
            

        sign90 : The index of MSSA PC that is above the 90% significance level
        sign95 : The index of MSSA PC that is above the 95% significance level
        sign97_5 : The index of MSSA PC that is above the 97.5% significance level
        sign99 : The index of MSSA PC that is above the 99% significance level
        
        eigenvalue_suro : numpy.ndarray
        The eigenvalue of the surrogates
        
        '''
        Ns = config.realizations

        if hasattr(self, 'ts_pc'):
            y = self.ts_pc # takes PCs from all variables stacked together
        else:
            y = self.ts
            
        N,D = y.shape
        M = self.M

        L = N - M+1
        #print(N)
        #print(D)
        #print(L)
        #print(self.M)

        N2 = max([L,D*M])        
        
        iin=0 # initiate loop        
        nn0 = [0]
        nn = []

        while (len(nn0)!=len(nn)) and (iin<3):              #  


            nn0 = np.copy(nn)
            
            alpha,gamma = tool.ar1fitcomptr(self.left_vectors,self.trajectory_matrix,n=config.norotate)   ### estimate alpha and gamma parameter to create red noise realizations. 
            #                                                                                       Norotate contains significant T-EOFs (?) that are excluded from the data that the red noise parameters are fit to.
            #                                                                                       Following AS1996 and GG2015 methodology regarding composite null-hypothesis
            #  for annotations, see tool.ar1fitcomptr
            
            r = np.zeros((N,D,Ns))
            
            R = np.zeros((N,D,Ns))
            for k in range(D):                                                                          # white noise realizations (R)
                for n in range(Ns):
                    temp=np.random.default_rng().normal(0, 1, size=(N))
                    R[:,k,n]=temp
            
            idx=hankel(np.arange(L),np.arange(N-M,N))
            for k in range(D):       
                r[:,k,:] = lfilter(np.array([np.sqrt(alpha[k])]),np.array([1,-gamma[k]]),R[:,k,:],axis=0)    # red noise realizations (r)
                                                                                                          ## square root of alpha is used to ensure the variance of red noise is correctly scaled.


            #TB - Here we apply the Procrustes target rotation ...
            if config.signif_test == SignTest.PROCRUSTES.value:
                Lambda_Procrustes = np.zeros((Ns,self.left_vectors.shape[1]))


                for n in tqdm(range(Ns)):                               
                    Yn = np.zeros((L,self.M,D))
                    for d in range(D):
                        xin = r[:,d,n]
                        Yn[:,:,d] = (xin[idx])
                    Yn = np.reshape(Yn.swapaxes(1,2),(L,D*M))
                    un,wn,vn = svd(Yn/np.sqrt(N2),full_matrices=False)                                 # Yn is the trajectory matrix of each surrogate (L, M, D) - Reshaped to 2D -> (L, D*M). Normalized by the maximum of L & D*M
                    vn = vn.T
                                                                                                    # These steps introduce the Procrustes rotation... formulas from Groth&Ghil 2015

                    pn = (un@np.diag(wn)).T@(self.left_vectors@np.diag(self.singularvalue))               # eq 10B (GG2015): E_r = U_n, Sigma_R = wn ), self.left_vectors (= data_left_singular_vectors = EU, self.singuularvalue = EW )
                                                                                                    # (E_r @ Sigma_R)^T @   E_data @ Sigma_data  == USV'
                    up,sp,vp = svd(pn)                                                              # eq 10b right hand side gives U S V' 

                    Te = np.real(up)@np.real(vp)                                                          # eq 10a  , thus

                    #Wn_wrong[n,:] = np.diag(Te.T@np.diag(wn)@Te)**2                                             # eq 11 (!) squared -> directly calculates eigenvalue instead of singular value (see below: self.eigenvalue_suro = Wn)
                    Lambda_Procrustes [n,:] = np.diag(Te.T@np.diag(wn**2)@Te)   
                    Wn = Lambda_Procrustes.copy()

            if (config.signif_test == SignTest.DATA_BASIS.value) | (config.signif_test == SignTest.NOISE_BASIS.value):    
                C_realizations = np.zeros((L,L,Ns)) #shape L, L, Ns

                idx_hankel = hankel(np.arange(L), np.arange(N-M,N))

                for i in range(Ns):
                    X= np.zeros((L,M,D),complex)
                    ts_2d = r[:,:,i]
                    for d in range(D):
                        slice_data = ts_2d[:,d]
                        X[:,:,d] = slice_data[idx_hankel]
                    X_2d = np.reshape(X.swapaxes(1,2),(N-M+1,D*M))

                    if M*D > L:  #rank deficient                       # GG 2015 eq 13 (= AR96 eq. in first section 2)
                        C_2d = (X_2d @ X_2d.T ) / (M*D) 
                    elif L > M*D: #full rank
                        C_2d = (X_2d.T @ X_2d) / L      # GG 2015 eq 1 (= AR96 eq. in first section 2)

                    C_realizations[:,:,i] = C_2d
                C_r_av =   C_realizations.mean(axis = 2)

                Lambda_r_av, E_r_av = np.linalg.eig(C_r_av) # Eq. for Cn in text above 8a GrothGhil 2015. E_r_av = E_n in GG15. Sort based on Lambda.
                idx_sort = np.argsort(Lambda_r_av)[::-1]
                Lambda_r_av = Lambda_r_av[idx_sort]
                E_r_av = E_r_av[:, idx_sort]


                Lambda_6 = np.zeros((Ns, self.left_vectors.shape[1])) 
                X_2d = np.reshape(self.trajectory_matrix.swapaxes(1,2),(N-M+1,D*M)) # -> to 2D
                if M*D > L:
                    E_d = self.left_vectors #Data left vectors = EU
                    C_data = (X_2d @ X_2d.T ) / (M*D)
                elif L > M*D:
                    E_d = self.right_vectors  # data right vectors = EV
                    C_data = (X_2d.T @ X_2d) / L

                if (config.signif_test == SignTest.DATA_BASIS.value):
                    for i in range(Ns):
                        Lambda_6[i,:] = (E_d.T @ C_realizations[:,:,i] @ E_d).diagonal()        # eq 6. in GG15 (or acually eq 14 GG15, eq6 is for full rank case. )
                        Wn = Lambda_6.copy()
                elif(config.signif_test == SignTest.NOISE_BASIS.value):
                    Lambda_8a = (E_r_av.T @ C_data @ E_r_av).diagonal() # eq 8a. GG15
                    self.Lambda_8a = Lambda_8a
                    self.F_noise_base = time_a.dominant_freq(E_r_av)

                    Lambda_8b = np.zeros((Ns,L)) #
                    for i in range(Ns):
                        Lambda_8b[i,:] = (E_r_av.T @ C_realizations[:,:,i] @ E_r_av).diagonal() # eq 8b. in GG15
                    Wn = Lambda_8b.copy()


            iin+=1 # iteration in while loop
            
            sign90=[]
            sign95=[]
            sign97_5=[]
            sign99=[]

            per_choice= np.asarray([1, 2.5, 5, 10, 50, 90, 95, 97.5, 99])  #TB-percentiles to calculate significance for
            self.per_choice = per_choice
            siglevel_val = np.zeros((len(per_choice),self.left_vectors.shape[1]))
            for i in range(len(per_choice)):
                for k in range(self.left_vectors.shape[1]):                        
                    wk = np.sort(Wn[:,k])
                    siglevel_val[i,k] = wk[int(Ns*per_choice[i]/100)]
                    if i ==5:
                        if self.eigenvalue[k] > siglevel_val[i,k]:        
                            sign90.append(k)
                    elif i ==6:
                        if self.eigenvalue[k] > siglevel_val[i,k]:        
                            sign95.append(k)
                    elif i ==7:
                        if self.eigenvalue[k] > siglevel_val[i,k]:        
                            sign97_5.append(k)
                    elif i ==8:
                        if self.eigenvalue[k] > siglevel_val[i,k]:        
                            sign99.append(k)
            '''            
            #Ongarde=[]                                                                           # determinat significant levels.
            siglevel_val = np.zeros(self.left_vectors.shape[1])
            for k in range(self.left_vectors.shape[1]):                        
                wk = np.sort(Wn[:,k])
                siglevel_val[k] = wk[int(Ns*siglevel/100)]
                if self.eigenvalue[k] > siglevel_val[k]:        
                    Ongarde.append(k)
                    # print(k)'''

            #nn=Ongarde
            #print(nn0,nn)
        #self.siglevel = siglevel
        self.sign90 = sign90
        self.sign95 = sign95
        self.sign97_5 = sign97_5
        self.sign99 = sign99

        #self.isig = Ongarde
        self.eigenvalue_suro = Wn
        self.siglevel_value = siglevel_val

    def storing_spectrum_data(self, config:MSSAConfig):
        
        """
        Stores output from the spectral analysis from MSSA in a nc file. \n
        Stored at config.output_directory+'MSSA_Spectra/'+config.get_filename()+'.nc'

        Stored data is used in Spectrum_plotter_config.
        """

        filePath = config.output_directory+'MSSA_Spectra/'+config.get_filename()+'.nc' 
        print("Storing Spectrum data in: " + filePath)
        if os.path.exists(filePath):
            os.remove(filePath)

        Spectra_data = netcdf.Dataset(filePath, 'w')

        #create dimensions
        Spectra_data.createDimension('freq', len(self.F))
        Spectra_data.createDimension('M', 1) # 
        Spectra_data.createDimension('L', int(self.L)) # 

        #creating variables
        Spectra_data.createVariable('data_power', float, ('freq'), zlib=True)
        Spectra_data.createVariable('freq', float, ('freq'), zlib=True)
        Spectra_data.createVariable('M', int, ('M'), zlib=True)
        Spectra_data.createVariable('T', float, ('freq'), zlib=True) # 

        Spectra_data.createVariable('sigma_T', float, ('freq'), zlib=True) # 
        #adding variables
        Spectra_data.variables['data_power'].long_name	= 'Power of the ST-EOFs'
        Spectra_data.variables['freq'].long_name	= 'Array of dominant frequencies'
        Spectra_data.variables['M'].long_name	= 'Window size'
        Spectra_data.variables['T'].long_name	= 'Dominant Periods based on the periodogram (inverse of F)'
        Spectra_data.variables['sigma_T'].long_name	= 'Uncertainty (1std) in dominant period estimation. see (Saraswati et al., 2023)'
        #Writing data to correct variable
        Spectra_data.variables['data_power'][:] 		= self.eigenvalue[:len(self.F)]

        if (config.signif_test == SignTest.NOISE_BASIS.value):
            Spectra_data.createVariable('Noise_base_power', float, ('L'), zlib=True)
            Spectra_data.variables['Noise_base_power'][:] 		= self.Lambda_8a
            Spectra_data.variables['Noise_base_power'].long_name = 	"Power of eigenvalues (Lambda_n) associated with the average covariance matrix Cn (eq 8a GG15))"

            Spectra_data.createVariable('F_noise_base', float, ('L'), zlib=True)
            Spectra_data.variables['F_noise_base'][:] 		= self.F_noise_base
            Spectra_data.variables['F_noise_base'].long_name = 	"dominant frequency of eigenbase associated with the average covariance matrix Cn (eq 8a GG15))"

        Spectra_data.variables['freq'][:]		= self.F
        Spectra_data.variables['M'][:]		= self.M
        Spectra_data.variables['T'][:]		        = self.T
        Spectra_data.variables['sigma_T'][:]		= self.sigma_T

        Spectra_data.createDimension('nn', len(config.norotate)) 
        Spectra_data.createDimension('per', len(self.per_choice)) 
        Spectra_data.createDimension('isign90', None) # 
        Spectra_data.createDimension('isign95', None) # 
        Spectra_data.createDimension('isign97_5', None) # 
        Spectra_data.createDimension('isign99', None) # 

        Spectra_data.createVariable('nn', int, ('nn'), zlib=True)
        Spectra_data.createVariable('per', float, ('per'), zlib=True)
        Spectra_data.createVariable('isign90', int, ('isign90'), zlib=True) # 
        Spectra_data.createVariable('isign95', int, ('isign95'), zlib=True) # 
        Spectra_data.createVariable('isign97_5', int, ('isign97_5'), zlib=True) # 
        Spectra_data.createVariable('isign99', int, ('isign99'), zlib=True) # 


        Spectra_data.variables['nn'].long_name	= 'Array of EOFs without nn rotation .. (?)'
        Spectra_data.variables['isign90'].long_name	= 'index of significant modes for upper CI at 90%'
        Spectra_data.variables['isign95'].long_name	= 'index of significant modes for upper CI at 95%'
        Spectra_data.variables['isign97_5'].long_name	= 'index of significant modes for upper CI at 97.5%'
        Spectra_data.variables['isign99'].long_name	= 'index of significant modes for upper CI at 99%'
        Spectra_data.variables['per'].long_name	= 'choice of significance percentage'


        #Writing data to correct variable
        if config.signif_test != SignTest.NOISE_BASIS.value:
            Spectra_data.createVariable('MC_power', float, ('per', 'freq'), zlib=True)
            Spectra_data.variables['MC_power'][:]		= self.siglevel_value[:,:len(self.F)]
        else:
            Spectra_data.createVariable('MC_power', float, ('per', 'L'), zlib=True)
            Spectra_data.variables['MC_power'][:]		= self.siglevel_value[:,:(self.L)]

        Spectra_data.variables['MC_power'].long_name	= 'Percentiles of Monte Carlo surrogates basis'

        Spectra_data.variables['per'][:]            = self.per_choice
        Spectra_data.variables['isign90'][:]		= self.sign90
        Spectra_data.variables['isign95'][:]		= self.sign95
        Spectra_data.variables['isign97_5'][:]		= self.sign97_5
        Spectra_data.variables['isign99'][:]		= self.sign99
        Spectra_data.variables['nn'][:]             = config.norotate
        Spectra_data.close()

    def storing_EEOF_data(self, config:MSSAConfig, list_data_std):
        '''        
        Stores output from the EEOFs in a nc file. \n
        Data stored in config.output_directory+'EEOFs/'+config.get_filename()+'.nc' 

        Stored data is used in EEOF_plotter_config or to retrieve RCs using functions in Retrieve_PC_RC.
        
        '''
        dim_i = self.dims[0]
        dim_j = self.dims[1]
        if config.level_zonal_bottom == LevelZonalBottom.Three_D.value:
            dim_k = self.dims[2]


        filePath = config.output_directory+'EEOFs/'+config.get_filename()+'.nc' 

        print("Storing ST-PCs in: " + filePath)  
        if os.path.exists(filePath):
            os.remove(filePath)
        
        
        EEOF_data = netcdf.Dataset(filePath, 'w')  #creates Dataset nc file on path location, in 'w' mode = writing access.
        ## creates dimensions, variables, etc.

        EEOF_data.createDimension('time', len(self.time))



        if ((config.level_zonal_bottom == LevelZonalBottom.LEVEL.value) | (config.level_zonal_bottom == LevelZonalBottom.BOTTOM.value)| (config.level_zonal_bottom == LevelZonalBottom.VOL.value)):
            EEOF_data.createDimension('lon', len(dim_i)) 
            EEOF_data.createDimension('lat', len(dim_j))

            EEOF_data.createVariable('lat', float, ('lat'), zlib=True)
            EEOF_data.createVariable('lon', float, ('lon'), zlib=True)
            EEOF_data.createVariable('masked_field', float, ('lat', 'lon'), zlib=True)

            EEOF_data.variables['lon'].long_name 		= 'Array of longitudes'
            EEOF_data.variables['lat'].long_name 		= 'Array of latitudes'

            EEOF_data.variables['lon'].units 		= 'Degrees E'
            EEOF_data.variables['lat'].units 		= 'Degrees N'

            EEOF_data.variables['lon'][:] 			= dim_i 
            EEOF_data.variables['lat'][:] 			= dim_j

        elif (config.level_zonal_bottom == LevelZonalBottom.ZONAL.value):
            EEOF_data.createDimension('lat', len(dim_i))
            EEOF_data.createDimension('depth', len(dim_j))

            EEOF_data.createVariable('lat', float, ('lat'), zlib=True)
            EEOF_data.createVariable('depth', float, ('depth'), zlib=True)
            EEOF_data.createVariable('masked_field', float, ('depth', 'lat'), zlib=True)
            
            EEOF_data.variables['depth'].long_name 		= 'Array of depth levels'
            EEOF_data.variables['depth'].units 		= 'meters depth'
            EEOF_data.variables['lat'].long_name 		= 'Array of latitudes'
            EEOF_data.variables['lat'].units 		= 'Degrees N'

            EEOF_data.variables['lat'][:] 			= dim_i
            EEOF_data.variables['depth'][:] 			= dim_j

        elif (config.level_zonal_bottom == LevelZonalBottom.Three_D.value):
            EEOF_data.createDimension('lon', len(dim_i)) 
            EEOF_data.createDimension('lat', len(dim_j)) 
            EEOF_data.createDimension('depth', len(dim_k))

            EEOF_data.createVariable('masked_field', float, ('depth', 'lat', 'lon'), zlib=True)
            EEOF_data.createVariable('depth', float, ('depth'), zlib=True)
            EEOF_data.createVariable('lat', float, ('lat'), zlib=True)
            EEOF_data.createVariable('lon', float, ('lon'), zlib=True)

            EEOF_data.variables['lon'].long_name 		= 'Array of longitudes'
            EEOF_data.variables['depth'].long_name 		= 'Array of depth levels'
            EEOF_data.variables['lat'].long_name 		= 'Array of latitudes'

            EEOF_data.variables['lon'].units 		= 'Degrees E'
            EEOF_data.variables['depth'].units 		= 'meters depth'
            EEOF_data.variables['lat'].units 		= 'Degrees N'

            EEOF_data.variables['lon'][:] 			= dim_i 
            EEOF_data.variables['lat'][:] 			= dim_j
            EEOF_data.variables['depth'][:] 			= dim_k

        EEOF_data.createDimension('channels', self.ts[0].shape[1]) # 1033, ie dim_i*dim_j - nans 
        EEOF_data.createDimension('input_pc', len(self.n_pc_list)) 
        EEOF_data.createDimension('output_eof', config.n_rec_comp) 
        EEOF_data.createDimension('L', self.L) 


        EEOF_data.createVariable('time', float, ('time'), zlib=True)


            
        EEOF_data.createVariable('input_pc', int, ('input_pc'), zlib=True)
        EEOF_data.createVariable('output_eof', float, ('output_eof'), zlib=True) #          here ST-EOF based on dimension u1 and u2 ... but Dim u1 based on shape u2 ? see comment above?
        #EEOF_data.createVariable('RCs', float, ('time', 'channels','output_eof'), zlib=True)
        EEOF_data.createVariable('ST_PCs', float, ('L', 'output_eof'), zlib=True)
        EEOF_data.createVariable('var_expl_mssa', float, ('L'), zlib=True)


        #EEOF_data.variables['data_all'].long_name 	= 'Normalised data time series'
        EEOF_data.variables['masked_field'].long_name 	= 'Normalised data time series - masked array to reconstruct RC into proper latxlon field'

        EEOF_data.variables['input_pc'].long_name 	= 'n_pc. Number of EOFs/PC used in the pre-PCA of variable x:' 
        EEOF_data.variables['output_eof'].long_name 	= 'n_rec. The number of the reconstructed components that is generated from the MSSA process.'

        #EEOF_data.variables['RCs'].long_name 	= 'RC_spatial. Normalised space time EOFs of variable' + config.variable
        #EEOF_data.variables['ST-SVs'].long_name 	= 'Space time singular values'
        EEOF_data.variables['ST_PCs'].long_name 	= 'space time PCs/EEOFs from the MSSA procedure'
        EEOF_data.variables['var_expl_mssa'].long_name 	= 'Variance explained by ST_PCs in %. Not cummulative.'


        EEOF_data.variables['time'].units 		= 'Years'

        #Writing data to correct variable

        EEOF_data.variables['time'][:]			= self.time

            
        #EEOF_data.variables['data_all'][:] 		= data_all
        EEOF_data.variables['masked_field'][:]		= self.masked_field
        EEOF_data.variables['input_pc'][:]		= self.n_pc_list
        EEOF_data.variables['output_eof'][:]		= config.n_rec_comp

        #EEOF_data.variables['RCs'][:]		= Rx               # here ST-EOFs consists only of u2 data
        #EEOF_data.variables['ST-SVs'][:] 		= s2
        EEOF_data.variables['ST_PCs'][:] 		= self.PC
        EEOF_data.variables['var_expl_mssa'][:] 		= self.var_expl_mssa

        for i in range(config.numberOfVar):
            RC_var_name_String = 'RCs_' + str(i+1)
            EEOF_data.createVariable(RC_var_name_String, float, ('time', 'channels','output_eof'), zlib=True)
            EEOF_data.variables[RC_var_name_String].long_name 	= 'RC_spatial. Normalised space time EOFs of variable' + config.get_list_variables()[i]
            EEOF_data.variables[RC_var_name_String][:]		= self.RC_spatial[i]              # here ST-EOFs consists only of u2 data

            std_var_name_String = 'std_var' + str(i+1)
            EEOF_data.createVariable(std_var_name_String, float, ('channels'), zlib=True)
            EEOF_data.variables[std_var_name_String].long_name 	= f"STD of the data for variable {config.get_list_variables()[i]} for each channel. Can be used to rescale RC from z-score into original unit of the original variable. "
            EEOF_data.variables[std_var_name_String][:]		= list_data_std[i]              # here ST-EOFs consists only of u2 data

            data_all_var_name_String = 'input_data_var' + str(i+1)
            EEOF_data.createVariable(data_all_var_name_String, float, ('time','channels'), zlib=True)
            EEOF_data.variables[data_all_var_name_String].long_name 	= f"Input data of the data for variable {config.get_list_variables()[i]} used in MSSA analysis. Mean is removed and is also standardised by std."
            EEOF_data.variables[data_all_var_name_String][:]		= self.ts[i]              # here ST-EOFs consists only of u2 data

    #def return_MC_99(self):



