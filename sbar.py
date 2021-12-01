import numpy as np
from numpy import pi, sin, log
from scipy.special import sici
from scipy.optimize import curve_fit

def obm_mean_var(signal, batch_len, batch_overlap):    
    """
    Divide signal into overlapping batches and get variance of batch means
        
    Parameters
    ----------
    signal : ndarray
        1D array 
    batch_len : int 
        length of batches
    batch_overlap : int 
        overlap of batches, should be between 0 and batch_len-1

    Returns
    -------
    float
        mean variance of signal at batch_length
        
    """    
    #stackoverflow.com/questions/13728392/moving-average-or-running-mean
    cumsum = np.cumsum(np.insert(signal, 0, 0)) 
    mavg = (cumsum[batch_len:] - cumsum[:-batch_len]) / float(batch_len)
    
    if batch_len <= batch_overlap:
        raise Exception('batch_overlap must be less than batch_len-1')
    step = batch_len - batch_overlap
    return np.var(mavg[::step])

def bm_mean_var(signal, batch_len):    
    """
    Divide signal into batches and get mean of batch variance
        
    Parameters
    ----------
    signal : ndarray
        1D array 
    batch_len : int 
        length of batches

    Returns
    -------
    float
        mean variance of signal at batch_length
        
    """    
    #trim rem off end of signal if length not divisible by # batches
    rem = signal.size%batch_len
    nb = int(signal.size/batch_len)    
    if(rem == 0):
        batches = signal.reshape(nb, batch_len)
    else:
        batches = signal[:-rem].reshape(nb, batch_len)
    return np.average(np.var(batches,1))


def bm_var_mean(signal, batch_len):   
    """
    Divide signal into batches and get the variance of the batch means
    
    Parameters
    ----------
    signal : ndarray
        1D array 
    batch_len : int 
        length of batches

    Returns
    -------
    float
        variance of mean of signal at batch_length
        
    """    
    
    #trim rem off end of signal if length not divisible by # batches
    rem = signal.size%batch_len 
    nb = int(signal.size/batch_len)
    if(rem == 0):
        batches = np.split(signal, nb)
    else:
        batches = np.split(signal[:-rem], nb)
    return np.var(np.average(batches,1))


def var_bl(t,B):
    """        
    Parameters
    ----------
    t : float
        time 
    B : float 
        bandwidth (1/s)

    Returns
    -------
    float
        expected variance/total variance of band-limited noise (goes to 1 at higher t)
    """
    return 1. - var_mean_bl(t,B)

def log_var_bl(t,B):
    """
    log(var_bl(t,B)), used for fitting
    """
    return np.log(1. - var_mean_bl(t,B))
    
def var_mean_bl(t, B):
    """        
    Parameters
    ----------
    t : float
        time 
    B : float 
        bandwidth (1/s)

    Returns
    -------
    float
        expected variance in mean of band-limited noise
        
    """
    return (-(sin(B*pi*t))**2 + B*pi*t*sici(2*B*pi*t)[0])/((B*pi*t)**2)

def log_var_mean_bl(t, B):
    """ 
    log(var_mean_bl(t,B)), used for fitting
    """
    return log(var_mean_bl(t,B))


def var_mean_bl_mean_fit(signal, vc=False, min_b = 20, fit_b = 10):
    """
    Get the estimated variance of the signal 
    
    Parameters
    ----------
    signal : ndarray
        1D array       
    vc : bool
        correct for bias in signal variance at early times
    min_b : int, optional
        minimum number of batches used when fitting BL function to batch means
    fit_b : int, optional
        number points to use when fitting BL function to batch means 
        
    
    Notes
    -----
    
    The method estimates the variance in the mean of a signal by calculating
    the variance in the mean vs time with the batch means method, then 
    extrapolating to the latest time assuming the scaling of bandwidth limited 
    noise
    """
    
    var_sig = np.var(signal)
    if(len(signal) < 20):
        return var_sig
    bl_end = int(signal.size/min_b)
    var_est = bm_mean_var(signal, bl_end)/var_sig
    B_est = 1./(2*var_est*bl_end)
    bl_start = 1./(2*B_est) 
            

    batches = np.logspace(np.log10(bl_start), np.log10(bl_end), fit_b).astype(int)
    batches = np.unique(batches[batches != 0])
    varb = np.zeros(len(batches))

    
    for i,b in enumerate(batches):
        varb[i] = bm_var_mean(signal, b)
    varb /= var_sig 
 
    B = curve_fit(log_var_mean_bl, xdata = batches, ydata = log(varb), 
                      p0 = B_est, bounds = (0, np.inf))[0][0]
    if vc:
        var_corr = var_bl(len(signal), B)
    else:
        var_corr = 1

    return var_mean_bl(len(signal), B)*var_sig/var_corr

def var_mean_bl_var_fit(signal, vc =False, min_bl = 20, fit_b = 10):
    """
    Get the estimated variance of the signal 
    
    Parameters
    ----------
    signal : ndarray
        1D array
    vc: bool
        correct for bias in signal variance at early times
    min_bl : int, optional
        minimum batch lenth used when fitting BL function to batch means
    fit_b : int, optional
        number points to use when fitting BL function to batch means 
        
    Notes
    -----
    
    The method estimates the variance in the mean of a signal by calculating
    the mean variance vs time with the batch means method, then extrapolating 
    to the latest time assuming the scaling of bandwidth limited noise
    """
    
    var_sig = np.var(signal)
    if(len(signal) < 20):
        return var_sig
    bl_end = int(signal.size/20)
    var_est = bm_mean_var(signal, bl_end)/var_sig
    B_est = 1./(2*var_est*bl_end)
    bl_end = 1./(2*B_est) 
    bl_start = min_bl
    
    batches = np.logspace(np.log10(bl_start), np.log10(bl_end), fit_b).astype(int)
    batches = np.unique(batches[batches > 1])
    varb = np.zeros(len(batches)) 

    for i,b in enumerate(batches):
        varb[i] = bm_mean_var(signal, b)    
    varb/=var_sig
    
    B = curve_fit(log_var_bl, xdata = batches, ydata = log(varb), p0 = B_est,
                  bounds = (0, np.inf))[0][0]
    if vc:
        var_corr = var_bl(len(signal), B)
    else:
        var_corr = 1
    return var_mean_bl(len(signal), B)*var_sig/var_corr