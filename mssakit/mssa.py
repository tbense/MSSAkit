import sys
import numpy as np
import scipy as sc
from numpy import *
from array import *
from numpy.linalg import *
from scipy.linalg import *
from matplotlib.pyplot import *
import matplotlib as mpl
from numpy.linalg import *
from scipy.interpolate import *
from tqdm import tqdm
from scipy.signal import lfilter,periodogram
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

from mssakit.pca import *
import mssakit.toolbox as tool
import mssakit.time_analysis as time

class mssa:
    '''Performs MSSA decomposition on multivariate time series.
    Multivariate time series should have observations in columns and time series
    indices in rows.
    
    Parameters
    ------------
    ts : numpy.ndarray : (time_steps, channels)
        Time series array.
        
    M : int | None
        The window size parameter controls the dimensionality of the trajectory
        matrices constructed for each timeseries (and then stacked). Timeseries
        are converted into trajectory matrices through "hankelization", where
        columns and rows represent different "windows" of the timeseries,
        incrementing across the timeseries. With window_size = M, the resulting
        trajectory matrix of a timeseries vector (N) will be of shape (M, K), where
        K = N - M + 1. As such, window_size should be no greater than N // 2. If
        left as None, MSSA will select N // 3.
        
   n_rec : int | None
        The number of the reconstructed components that is generated from the MSSA process. The default is 50.
        
    pre_pca : boolean
        PCA is applied to the data before applying MSSA to reduce its dimension. The default is False.
    
    n_pc : int | None
        The number of the PCs that is kept and used for the MSSA process. The default is all PCs.
    
    varimax : boolean
        The option to apply varimax rotation in MSSA. The default is False.
    
    t : numpy.array (optional)
        The time indicator of array ts.
        
    sampling_interval : int
        The sampling interval of the time series. The default is 1.
        
    
    Attributes
    ----------
    ts : numpy.ndarray
        The input of the time series.     
    
    pca_PC : numpy.ndarray
        The set of PCs that is selected from the pre-PCA, if pre_pca is selected.
    
    pca_u : The set of left singular vectors that is selected from the pre-PCA.

    pca_v : The set of right singular vectors that is selected from the pre-PCA.

    pca_s : The set of singular values that is selected from the pre-PCA.

    Eigenvalue = pc_s**2, var_expl = Eigenvalue / sum(Eigenvalue). 
    pca_RC = usv: RC = pc_u[:,:n_components] @ diag(pc_s[:n_components]) @ transpose(pc_v[:,:n_components])

    t : array
        The given time index of the time series. If t is not given, the time index will
        be replaced with the i-index point.
    
    sampling_interval : int
        The given sampling interval of the time series.
    
    M : int
        The given windows size used in the MSSA.        
    
    trajectory_matrix : numpy.ndarray
        The resulting trajectory matrix in the MSSA
    
    left_vectors : numpy.ndarray
        The resulting left singular vectors from SVD of the trajectory matrix, also familiar with Time-EOF.
    
    right_vectors : numpy.ndarray
        The resulting right singular vectors from SVD of the trajectory matrix, also familiar with SpaceTime-EOF.
        
    singularvalue : numpy.array
        The resulting singular values from SVD of the trajectory matrix.
    
    eigenvalue : numpy.array
        The eigenvalues of the trajectory matrix (singularvalue ** 2).
    
    mssa_PC : numpy.ndarray
        The Time-PC from the MSSA.
    
    mssa_RC : numpy.ndarray
        The reconstruction of MSSA in the original multi-channel data
        (or PCA components if the pre_pca is selected). The dimension of this 
        matrix is (N, D, rank), where N is the number of observations, D is 
        the number of channels, and rank is the number of components selected to keep.
    
    mssa_RC_spatial : numpy.ndarray
        The reconstruction of MSSA in the original multi-channel data, if the pre_pca is selected.
        The dimension of this matrix is (N, D, rank), where N is the number
        of observations, D is the number of channels, and rank is the
        number of components selected to keep.
    
    F : numpy.array
        The dominant frequency of the T-EOF.
    
    T : numpy.array
        The dominant period of the T-EOF.
    
    sigma_T : numpy.array
        The period uncertainty of the obtained dominant period.
    '''       
    


    def __init__(self,
                 ts,                 
                 M=None,
                 n_rec=30,
                 pre_pca=False,
                 n_pc=None,
                 varimax=False,
                 t=None,
                 sampling_interval=1):
        
        self.ts = ts #data (time, channels)
        self.t = squeeze(t) #time_values
        self.sampling_interval = sampling_interval
        
        if pre_pca: # perform initial PCA, used as input to MSSA
            ts_pca = pca(ts)
            pc = ts_pca.PC
            
            if n_pc is None:
                n_pc = pc.shape[0]            
            y = pc[:,:n_pc]
            vr = ts_pca.v[:,:n_pc]
            
            self.pca_PC = copy(y) 
            self.pca_u = copy(ts_pca.u[:,:n_pc])
            self.pca_v = copy(vr)
            self.pca_s = ts_pca.s[:n_pc]
            self.pca_n_pc = n_pc

        else:
            y = copy(ts)
        
        if M is None:
            M = y.shape[0] // 3

        N,D = y.shape
        for k in range(D): # remove mean from each y
            y[:,k]-=mean(y[:,k])
        
        self.M = M    # Window size
        self.N = N    #time steps
        self.D = D    #Channels
        L = int(N-M+1)     # Adjusted window size N'=L=N-M+1
        self.L = L
        
        # creating trajectory matrix
        idx = hankel(arange(L),arange(N-M,N)) 
        xtde = zeros((L,M,D),complex)           # trajectory matrix X 
        for d in range(D):
            xin = y[:,d]    
            xtde[:,:,d] = xin[idx]            
        self.trajectory_matrix = xtde
        
        xtde = reshape(xtde.swapaxes(1,2),(N-M+1,D*M)) # reshape trajectory matrix into X (L, M*D) , ie by stacking the lagged copies of the matrix behind each other
        
        #  perform MSSA by applying SVD to the trajectory matrix, devided by sqrt(D*M) to make the singular values squared compare directly to variance explained.
        #  Applying SVD directly on trajectory matrix X gives us the singular vectors (left, EU and right EV^T) which represent T-EOFs and ST-EOFs as well as the singular values Lamda_D.
        #  right singular vectors EV=E_d . (eq 4 AS96), eq2 GG15

        EU,EW,EV = svd(xtde/sqrt(D*M),full_matrices=False)
        EV = EV.T
        
        # rank sort
        ks = EW.argsort()[::-1]
        EW = EW[ks]
        EV = real(EV[:,ks])
        EU = real(EU[:,ks])
        eigval = EW**2          # eigenvalues = singular values squared.
        
        if varimax:            
            # Rotate only the components with % of eigenvalues > 1%
            Sm = nonzero(eigval/sum(eigval)>0.01)[0][-1]+1              # select the index Sm where % explained>= 1%
            
            EVsc = EV[:,:Sm]@diag(EW[:Sm])                              # cross product eigenvectors EV with singular values to scale them appropriately
            EVr = zeros((M,D,Sm))                                       # create empty arrar of size M,D3,Sm and place values of EVsc in there
            for k in range(Sm):
                EVr[:,:,k] = reshape(EVsc[:,k],(D,M)).T                 # EVr corresponds to S in Groth and Ghil 2011
                
            dummy,Tv = tool.varimax_fun(real(EVr))                      # Apply varimax rotation, see toolbox.varimax_fun for annotations. Also Groth & Ghil 2011. 
                                                                        # Tv = Rotation matrix T from eq. 9 in (GG2011)
            EV0 = copy(EV)
            EU0 = copy(EU)
            EV[:,:Sm] = EV[:,:Sm] @ Tv                                  # Update the first Sm eigenvectors (non normalised = EV) by applying formula 9 (GG11) 
            EW[:Sm] = diag(Tv.T@diag(EW[:Sm])@Tv)                       # Formula 12 (GG11)
            EU[:,:Sm] = EU[:,:Sm]@Tv
    
    
            # sort again EW
            ks = EW.argsort()[::-1]
            EW = real(EW[ks])
            EV = EV[:,ks]
            EU = EU[:,ks]            
            eigval = EW**2
    
            EV = real(EV)
            EU = real(EU)

            ll = arange(0,D*M,M)

            Xv = (EU@diag(EW)@EV.T)*sqrt(D*M)                           #recompute the trajectory matrix by a reverse SVD and reversing the normalisation (Why D*M)
            Xtdev=reshape(Xv,(L,D,M)).swapaxes(2,1)                     # shape trajectory matrix X to 3D (L, M,D)
            yv=Xv[:,ll]
            yvv=Xv[L-M+1:,ll+M-1]            
            yv=concatenate((yv,yvv),axis=0)            
    
            ew=diag(EW)
            
            self.trajectory_matrix = Xtdev                              # overwrite trajectory matrix with new one after varimax.
            
        self.left_vectors = EU
        self.right_vectors = EV
        self.singularvalue = EW
        self.eigenvalue = eigval
        self.var_expl_mssa =  np.around((self.eigenvalue / self.eigenvalue.sum() * 100),3) # store amount of variance that is explained by each ST-PCs

        # % get V and R in all pc
        NC = n_rec
        FF=zeros(NC)
        a=zeros((N,D))
        A=zeros((L,NC));
        RC=zeros((N,D,NC))
        Ru=zeros((N,D,NC))
        ia=0

        #The below loop implements eq 3 and 4 from GG2011 
        #a = pc = sum over D and M,  e_dk * x_d.  aka lfilter Ej * yr (normalised by 1 = not normalised)
        # RC = sum over D and M (implicit in hstack pc)
        # RC = a_k * e_dk   aka lfilter pc * Ej, normalised by M (see 1/M before the summation in eq4)

        for o in range(NC) :
            Ej=(EV[:,o].reshape((D,-1))).T
            for j in range(D):
                a[:,j]=lfilter(flip(Ej[:,j]),1,y[:,j])
            pc=sum(a,axis=1)[M-1:]
            A[:,ia]=pc
            pc=hstack((pc,zeros(M-1)))
            for j in range(D):                   
                RC[:,j,ia]=lfilter(Ej[:,j],M,pc)                #normalisation with factor M (ie * 1/M)    (text under eq. 52 Ghil et al. 2002)  
            ia+=1    
        for i in range(0,M-1):                                  ## some sort of edge corretion for the sliding window ->  (eq 52 -> 11/12 G2002)
            RC[i,:,:]=RC[i,:,:]*M/(i+1)
            RC[N-i-1,:,:]=RC[N-i-1,:,:]*M/(i+1)
            
            
        self.mssa_PC = A             # 2 / 3 in Groth&Ghil 2011
        self.mssa_RC = RC            # 4 GG2011 
        
        if pre_pca:
            Rx = zeros((ts.shape[0],ts.shape[1],NC))      #Convert Reconstructed components from current form (like the pre-pca output) to the input before pre-pca (ie, (time, channels, RCs))
            vt = transpose(vr)
            for i in range(NC):                
                Rx[:, :,i]=(RC[:,:,i] @ vt)  #/w*stdx    
                
            self.mssa_RC_spatial = Rx
        #Compute freq stats from eigenvectors instead of the MSSA PCs
        #self.F = time.dominant_freq(A,fs=sampling_interval)
        #self.T, self.sigma_T = time.dominant_period(A,fs=sampling_interval,uncertainty=True)
        if M*D >L: #left vectors are eigenbase - > EU
                self.F = time.dominant_freq(EU,fs=sampling_interval) 
                self.T, self.sigma_T = time.dominant_period(EU,fs=sampling_interval,uncertainty=True)

        elif L > M*D: #right vectors form eigenbase -> EV
                self.F = time.dominant_freq(EV,fs=sampling_interval) 
                self.T, self.sigma_T = time.dominant_period(EV,fs=sampling_interval,uncertainty=True)        
       
        
        
        
    def mcmssa(self, Ns=100, norotate=[]):        
        ''' Applying Monte Carlo to the MSSA
                
        Parameters
        ------------    
        Ns : int
            Number of surrogates. The default is 100.
        
        norotate : list
            The list of index of non-rotated T-EOF. The default is []
               

        Atributes
        ---------
        Ns : surrogates
        
        n_sign_95 : number of significant st-pcs above 95% percentile

        siglevel_value : array, shape (sign_perct, mssa_modes) -> contain the surrogate eigenvalue at the percentages in sign_use to plot vertical error bars in spectral plots. 

        self.sign_mask : array, shape (sign_perct, mssa_modes) -> True if significant at given level
        
        eigenvalue_suro : numpy.ndarray
            The eigenvalue of the surrogates
        
        '''
        if hasattr(self, 'pca_PC'):
            y = self.pca_PC
        else:
            y = self.ts
        self.Ns = Ns # retrievable attribute    
        N,D = y.shape

        M = self.M
        L = N - M+1
        N2 = max(L,D*M)        
        
        iin=0 # initiate loop        
        nn0 = [0]
        nn = []

        while (len(nn0)!=len(nn)) and (iin<3):            
            nn0 = copy(nn)
            
            alpha,gamma = tool.ar1fitcomptr(self.left_vectors,self.trajectory_matrix,n=norotate)   ### estimate alpha and gamma parameter to create red noise realizations. 
            #                                                                                       Norotate contains significant T-EOFs (?) that are excluded from the data that the red noise parameters are fit to.
            #                                                                                       Following AS1996 and GG2015 methodology regarding composite null-hypothesis
            #  for annotations, see tool.ar1fitcomptr            
            r = np.zeros((N,D,Ns))
            #Wn = zeros((Ns,self.left_vectors.shape[1]))
            R = np.zeros((N,D,Ns))
            for k in range(D):                                                                          # white noise realizations (R)
                for n in range(Ns):
                    temp=np.random.default_rng().normal(0, 1, size=(N))
                    R[:,k,n]=temp
            
            idx=hankel(np.arange(L),np.arange(N-M,N))
            for k in range(D):       
                r[:,k,:] = lfilter(np.array([np.sqrt(alpha[k])]),np.array([1,-gamma[k]]),R[:,k,:],axis=0)    # red noise realizations (r)
                                                                                                          ## square root of alpha is used to ensure the variance of red noise is correctly scale      
            
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
            iin+=1
            per_choice = np.asarray([1, 2.5, 5, 10, 50, 90, 95, 97.5, 99])
            self.per_choice = per_choice

            n_levels = len(per_choice)          # 9
            K = self.eigenvalue.size            # = left_vectors.shape[1]

            # --- significance thresholds, vectorised over modes -------------------
            Wn_sorted = np.sort(Wn, axis=0)                              # (Ns, K)
            idx = np.minimum((Ns * per_choice / 100).astype(int), Ns - 1)  # clamp
            siglevel_val = Wn_sorted[idx, :]                             # (n_levels, K)

            # --- boolean significance mask ---------------------------------------
            # sign_mask[i, k] is True  <=>  eigenvalue[k] > siglevel_val[i, k]
            sign_mask = self.eigenvalue[None, :] > siglevel_val           # (n_levels, K) bool

            # --- single headline count -------------------------------------------
            i95 = int(np.where(per_choice == 95)[0][0])
            n_sign_95 = int(sign_mask[i95].sum())

            self.siglevel_value = siglevel_val
            self.sign_mask = sign_mask
            self.n_sign_95 = n_sign_95

            #self.eigenvalue_suro = Wn don't actually store the eigenvalues of the surrogate, takes loads of space and is not needed

        #self.sign95 = _modes_above(sign_mask, per_choice, 95)   # same values, same order


            '''
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
            Ongarde=[]
            siglevel_val = zeros(self.left_vectors.shape[1])
            for k in range(self.left_vectors.shape[1]):                        
                wk = sort(Wn[:,k])
                siglevel_val[k] = wk[int(Ns*siglevel/100)]
                if self.eigenvalue[k] > siglevel_val[k]:        
                    Ongarde.append(k)
                    # print(k)
            nn=Ongarde
            print(nn0,nn)'''
           
        #self.siglevel = siglevel
        #self.isig = Ongarde
        #self.sign90 = np.array(sign90)
        #self.sign95 = np.array(sign95)
        #self.sign97_5 = np.array(sign97_5)
        #self.sign99 = np.array(sign99)

        
    def _modes_above(self, level):
                # e.g., self.sign95 = _modes_above(sign_mask, per_choice, 95)   # same values, same order
                return np.flatnonzero(self.sign_mask[int(np.where(self.per_choice == level)[0][0])])



    def get_group(self,f_threshold=0.01,comp='isig'):
        '''The function to group the T-EOF based on the similarity of the
        dominant frequency of each component.
        

        Parameters
        ----------
        f_treshold : float, optional
            The threshold of the frequencies that are in the same group of mode.
            The default is 0.01.
            
        comp : string | 'isig' | 'all'
            The function will only group the significant components if 'isig' is given.
            If 'all' is given, all components will be analyzed. The default is 'isig'.


        Returns
        -------
        lii : list
            The index of the components in the same group.
        Fli : array
            The average frequency of each group.

        '''
        
        if comp=='isig':
            idx = self.isig
        elif comp=='all':
            idx = list(arange(len(self.singularvalue)))
        else:            
            raise TypeError(f"Component error, choose 'isig' or 'all'")        
            return None
        
        Fo = self.F[idx]
        lii=[]
        lit=[]
        Fli=[]
        
        for k, value in enumerate(idx):
            Fi=nonzero((abs(Fo[k]-Fo[k+1:])<f_threshold))[0]            
            if (len(Fi)==0) & (k not in lit):                
                lii.append([value])
                Fli.append(Fo[k])
            elif (len(Fi)==1):
                lii.append([value,idx[k+1+Fi[0]]])
                Fli.append(mean([Fo[k],Fo[k+1+Fi[0]]]))
                lit.append(k+1+Fi[0])
            elif (k in lit):
                rien=0
            else:
                pass
        
        # self.group = lii
        # self.group_freq = Fli
        return lii, Fli
    
    
    def plot_mcmssa(self, ncomp=50):
        '''To plot the results of the Monte Carlo test of the MSSA.

        Parameters
        ----------
        ncomp : int, optional
            The first ncomp-rank of eigenvalues that are plotted. The default is 50.

        '''
        Ns = self.eigenvalue_suro.shape[0]
        top_level = int(Ns * (self.siglevel / 100))
        low_level = int(Ns * (1 - (self.siglevel / 100)))
        
        wk = sort(self.eigenvalue_suro, axis=0)
        
        tool.plot_mc(self.F, self.eigenvalue, wk, top_level, low_level, self.isig, ncomp)       