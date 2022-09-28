import numpy as np 
from scipy.stats import sem
from scipy import signal
from scipy import sparse
import cvxpy as cvx




#######################################



#################################


def get_coords(XYZ, mask=None, xlim_min = None, xlim_max= None, ylim_min= None,
               ylim_max= None, zlim_min= None,  zlim_max = None):
    
    
    
    xx =  np.squeeze(XYZ[:,0])
    yy =  np.squeeze(XYZ[:,1])
    zz =  np.squeeze(XYZ[:,2])

       
    new_mask = np.empty(XYZ.shape[0]) 
    new_mask[:] = True
    
    new_mask *= mask  if mask is not None else new_mask
    new_mask *= (xx >  xlim_min)  if xlim_min is not None else new_mask
    new_mask *= (xx <  xlim_max)  if xlim_max is not None else new_mask
    new_mask *= (yy >  ylim_min)  if ylim_min is not None else new_mask
    new_mask *= (yy <  ylim_max)  if ylim_max is not None else new_mask
    new_mask *= (zz >  zlim_min)  if zlim_min is not None else new_mask
    new_mask *= (zz <  zlim_max)  if zlim_max is not None else new_mask
    
    
    return np.argwhere(new_mask)




#####################################################

def get_stack(TC, central_tp, indx = None, left_lag=0, right_lag = 0):
    
    
    if indx is not None:

        TC_stack  = np.empty([len(central_tp),len(indx), int(left_lag+right_lag)]) 
        TC_stack[:] = np.NaN


        for tp in range(len(central_tp)):
            if (central_tp[tp] + right_lag <= TC.shape[1]): 
                TC_stack[tp,:,int(left_lag):int(left_lag+right_lag)] = np.squeeze(TC[indx,int(central_tp[tp]):int(central_tp[tp] + right_lag )])
            if (central_tp[tp] - left_lag >= 0): 
                TC_stack[tp,:,:int(left_lag)] = np.squeeze(TC[indx,int(central_tp[tp] - left_lag):int(central_tp[tp])])
                
    else:
        TC_stack  = np.empty([len(central_tp),TC.shape[0], int(left_lag+right_lag)]) 
        TC_stack[:] = np.NaN


        for tp in range(len(central_tp)):
            if (central_tp[tp] + right_lag <= TC.shape[1]): 
                TC_stack[tp,:,int(left_lag):int(left_lag+right_lag)] = np.squeeze(TC[:,int(central_tp[tp]):int(central_tp[tp] + right_lag )])
            if (central_tp[tp] - left_lag >= 0): 
                TC_stack[tp,:,:int(left_lag)] = np.squeeze(TC[:,int(central_tp[tp] - left_lag):int(central_tp[tp])])
                       
            
    return TC_stack


#########################

def compute_statistics(TC):
    
    mean_TC = np.nanmean(TC,0)
    std_TC = np.nanstd(TC,0)
    sem_TC = sem(TC,0, nan_policy='omit')
    
    return mean_TC, std_TC, sem_TC




#################################

def get_trials_latencyrange(trial_latency,latency_vector_min = None,latency_vector_max = None):
    
    lat_mask = np.empty(len(trial_latency))
    lat_mask[:] = True
     
    lat_mask *= (trial_latency <= latency_vector_max)  if latency_vector_max is not None else lat_mask
    lat_mask *= (trial_latency >= latency_vector_min)  if latency_vector_min is not None else lat_mask
    
    return(np.squeeze(np.argwhere(lat_mask)))


#################################
### modified from an already existig lab function to get unconvolved regressors


def get_bout_regressor(exp, time):
    bout_reg = np.zeros(exp.behavior_log.t.shape)
    bout_prop = exp.get_bout_properties()
    bouting_range =[ix for s, e
                       in zip(np.where(np.isin(exp.behavior_log.t, bout_prop["t_start"]))[0],
                           np.where(np.isin(exp.behavior_log.t, bout_prop["t_start"]+bout_prop["duration"]))[0])
                       for ix in range(s, e)]
    bout_reg[bouting_range] = 1

    # 6s kernel
    beh_freq = (exp.behavior_log.t[len(exp.behavior_log.t)-1]-exp.behavior_log.t[0])/len(exp.behavior_log.t)
    decay = np.exp(-np.arange(len(exp.behavior_log.t))*beh_freq / (1.5 / np.log(2)))
    kernel = decay / np.sum(decay)

    convolved = signal.convolve(bout_reg, kernel)
    convolved = convolved[np.searchsorted(exp.behavior_log.t, time)]
    unconvolved = bout_reg[np.searchsorted(exp.behavior_log.t, time)]
    return convolved, unconvolved



def GetSn(y, range_ff=[0.25, 0.5], method='mean'):
    """
    Estimate noise power through the power spectral density over the range of large frequencies
    Parameters
    ----------
    y : array, shape (T,)
        One dimensional array containing the fluorescence intensities with
        one entry per time-bin.
    range_ff : (1,2) array, nonnegative, max value <= 0.5
        range of frequency (x Nyquist rate) over which the spectrum is averaged
    method : string, optional, default 'mean'
        method of averaging: Mean, median, exponentiated mean of logvalues
    Returns
    -------
    sn : noise standard deviation
    """

    ff, Pxx = signal.welch(y)
    ind1 = ff > range_ff[0]
    ind2 = ff < range_ff[1]
    ind = np.logical_and(ind1, ind2)
    Pxx_ind = Pxx[ind]
    sn = {
        'mean': lambda Pxx_ind: np.sqrt(np.mean(Pxx_ind / 2)),
        'median': lambda Pxx_ind: np.sqrt(np.median(Pxx_ind / 2)),
        'logmexp': lambda Pxx_ind: np.sqrt(np.exp(np.mean(np.log(Pxx_ind / 2))))
    }[method](Pxx_ind)

    return sn


def cal_deconvolve(signals, calresp, method = 'spike', solver='ECOS'):
   
    T = signals.shape[0]
      
    if  method  == 'spike': 
        g = calresp
        G = sparse.dia_matrix((np.ones((1, T)), [0]), (T, T))
        for i, gi in enumerate(g):
            G  = G + sparse.dia_matrix((-gi * np.ones((1, T)), [-1 - i]), (T, T))
        
        denoised = np.zeros(signals.shape)
        deconvolved = np.zeros(signals.shape)
    
        for i in range(signals.shape[1]):
            y = signals[:,i] 
            c = cvx.Variable(T)  # calcium at each time step
            # objective = cvx.Minimize(.5 * cvx.sum_squares(c - y) + lam * cvx.norm(G * c, 1))
            # cvxpy had sometime trouble to find above solution for G*c, therefore

            lam =1/GetSn(y)
            b = cvx.Variable(1)
            objective = cvx.Minimize(.5 * cvx.sum_squares(c - y) +
                                    lam  * cvx.norm(G * c, 1))
            constraints = [G * c >= 0]

            prob = cvx.Problem(objective,constraints)
            prob.solve(solver=solver)
            deconvolved[:,i] = np.squeeze(np.asarray(G * c.value))
            denoised[:,i] = np.squeeze(np.asarray(c.value))

        
    if  method  == 'burst': 
        filt = np.zeros(len(calresp)+2) 
        filt[0] = 1
        filt[-1] = 0
        filt[1:-1] = calresp
        filt = -np.diff(filt)
        g = filt.tolist()
        
        
        G = sparse.dia_matrix((np.ones((1, T)), [0]), (T, T))
        for i, gi in enumerate(g):
            G  = G + sparse.dia_matrix((-gi * np.ones((1, T)), [-1 - i]), (T, T))
        
        denoised = np.zeros(signals.shape)
        deconvolved = np.zeros(signals.shape)
    
        for i in range(signals.shape[1]):
            y = signals[:,i] 
            c = cvx.Variable(T)  # calcium at each time step
            # objective = cvx.Minimize(.5 * cvx.sum_squares(c - y) + lam * cvx.norm(G * c, 1))
            # cvxpy had sometime trouble to find above solution for G*c, therefore

            lam =1/GetSn(y)
            b = cvx.Variable(1)
            objective = cvx.Minimize(.5 * cvx.sum_squares(c - y) +
                                    lam  * cvx.norm(G * c, 1))

            prob = cvx.Problem(objective)
            prob.solve(solver=solver)
            deconvolved[:,i] = np.squeeze(np.asarray(G * c.value))
            denoised[:,i] = np.squeeze(np.asarray(c.value))
        
    return denoised, deconvolved


        
def cal_deconvolve_1D(y, T , g , solver='ECOS'):
    
    G = sparse.dia_matrix((np.ones((1, T)), [0]), (T, T))
    for i, gi in enumerate(g):
        G  = G + sparse.dia_matrix((-gi * np.ones((1, T)), [-1 - i]), (T, T))
        
    
    
    c = cvx.Variable(T)  # calcium at each time step


    lam =1/GetSn(y)
    #b = cvx.Variable(1)
    objective = cvx.Minimize(.5 * cvx.sum_squares(c - y) +
                                lam  * cvx.norm(G * c, 1))
    constraints = [G * c >= 0]

    prob = cvx.Problem(objective)
    prob.solve(solver=solver)


    return np.squeeze(np.asarray(c.value))
        
        
        