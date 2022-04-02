import numpy as np

def noise_from_pow_spec(rfft_freqs, pow_spec, seed=None):
    """
    Generate noise with a given power spectrum.
    
    Parameters
    ----------
    rfft_freqs : ndarray
        1D array of real frequencies of noise to be generated
    power_spec : ndarray
        1D array of power spetrum of noise to be generated
    seed : int, optional
        seed for random number generation
    
    Returns
    -------
    ndarray
        1D array with noise with specified power spectrum
    
    Example
    -------
        
    >>>rfft = np.fft.rfft(noise)
    >>>rfft_freqs = np.fft.rfftfreq(noise.size, dt)
    >>>pow_spec = rfft*rfft.conj()*4*dt/(noise.size)
    >>>rand_noise = noise_from_power_spec(rfft_freqs, pow_spec) 
    """

    if seed is not None:
        np.random.seed(seed)
    rand_phase = np.array(np.sqrt(pow_spec), dtype='complex')
    phi = 2*np.pi*np.random.rand(rfft_freqs.size)  # pick a random phase angle for each freq
    phi = np.cos(phi) + 1j*np.sin(phi)
    rand_phase *= phi
    return np.fft.irfft(rand_phase).T


def band_limited_noise(f_c, bw, length, dt, mag_std=1., seed=None):
    """
    Generate bandwitdh limited white noise 
    
    Parameters
    ----------
    f_c : float
        central frequency of noise (1/s)
    bw : float
        band width of noise (1/s)
    length : float 
        length of signal (will be rounded down to nearest dt)(s)
    dt : float
        time step of signal (s)
    mag_std : float, optional
        standard deviation of returned signal
    seed : int, optional
        seed for random number generation

    Returns
    -------
    ndarray
        1D array of noise
    """
    min_freq = f_c - 0.5*bw
    max_freq = f_c + 0.5*bw
    sig_len = int(length/dt)

    freqs = np.fft.rfftfreq(sig_len, dt)
    p_spec = np.zeros_like(freqs)

    mask = (freqs > min_freq) & (freqs < max_freq)
    p_spec[mask] = 1.0
    
    df = freqs[1] - freqs[0]
    norm = dt*np.sqrt(2*df*np.sum(p_spec))/np.sqrt(length)

    
    return mag_std*noise_from_pow_spec(freqs, p_spec, seed=seed)/norm

def colored_noise(color, length, dt, mag_std, seed=None):
    B = 0
    if type(color) == float:
        B = color
    if type(color) == str:
        lcolor = color.lower()
        if lcolor == 'pink':
            B = -1
        elif lcolor == 'brown':
            B = -2
        elif lcolor == 'blue':
            B = 1
        elif lcolor == 'violet':
            B = 2
        elif lcolor == 'white':
            B = 0
            
    sig_len = int(length/dt)

    
    freqs = np.fft.rfftfreq(sig_len, dt)
    #print(np.fft.fftfreq(sig_len, dt))
    p_spec = np.power(freqs, B)
    #print(freqs, p_spec)
    #import matplotlib
    #import matplotlib.pyplot as plt
    #plt.plot(freqs ,p_spec, ".")
    #plt.show()
    p_spec[0] = 0
    #p_spec[-1] *= 0.5
    #w = p_spec.copy()
    p_spec[-1] *= (1 + (sig_len % 2)) / 2. 
    
     
    df = freqs[1] - freqs[0]
    print(df)
    norm = dt*np.sqrt(2*df*np.sum(p_spec))/np.sqrt(length)

    
    return mag_std*noise_from_pow_spec(freqs, p_spec, seed=seed)/norm


def powerlaw_psd_gaussian(exponent, size, fmin=0):
    """Gaussian (1/f)**beta noise.
    Based on the algorithm in:
    Timmer, J. and Koenig, M.:
    On generating power law noise.
    Astron. Astrophys. 300, 707-710 (1995)
    Normalised to unit variance
    Parameters:
    -----------
    exponent : float
        The power-spectrum of the generated noise is proportional to
        S(f) = (1 / f)**beta
        flicker / pink noise:   exponent beta = 1
        brown noise:            exponent beta = 2
        Furthermore, the autocorrelation decays proportional to lag**-gamma
        with gamma = 1 - beta for 0 < beta < 1.
        There may be finite-size issues for beta close to one.
    shape : int or iterable
        The output has the given shape, and the desired power spectrum in
        the last coordinate. That is, the last dimension is taken as time,
        and all other components are independent.
    fmin : float, optional
        Low-frequency cutoff.
        Default: 0 corresponds to original paper. It is not actually
        zero, but 1/samples.
    Returns
    -------
    out : array
        The samples.
    Examples:
    ---------
    # generate 1/f noise == pink noise == flicker noise
    >>> import colorednoise as cn
    >>> y = cn.powerlaw_psd_gaussian(1, 5)
    """
    
    # Make sure size is a list so we can iterate it and assign to it.
    try:
        size = list(size)
    except TypeError:
        size = [size]
    
    # The number of samples in each time series
    samples = size[-1]
    
    # Calculate Frequencies (we asume a sample rate of one)
    # Use fft functions for real output (-> hermitian spectrum)
    f = rfftfreq(samples)
    
    # Build scaling factors for all frequencies
    s_scale = f
    fmin = max(fmin, 1./samples) # Low frequency cutoff
    ix   = np.sum(s_scale < fmin)   # Index of the cutoff
    if ix and ix < len(s_scale):
        s_scale[:ix] = s_scale[ix]
    s_scale = s_scale**(-exponent/2.)
    
    # Calculate theoretical output standard deviation from scaling
    w      = s_scale[1:].copy()
    w[-1] *= (1 + (samples % 2)) / 2. # correct f = +-0.5
    sigma = 2 * sqrt(np.sum(w**2)) / samples
    
    # Adjust size to generate one Fourier component per frequency
    size[-1] = len(f)

    # Add empty dimension(s) to broadcast s_scale along last
    # dimension of generated random power + phase (below)
    dims_to_add = len(size) - 1
    s_scale     = s_scale[(newaxis,) * dims_to_add + (Ellipsis,)]
    
    # Generate scaled random power + phase
    sr = normal(scale=s_scale, size=size)
    si = normal(scale=s_scale, size=size)
    
    # If the signal length is even, frequencies +/- 0.5 are equal
    # so the coefficient must be real.
    if not (samples % 2): si[...,-1] = 0
    
    # Regardless of signal length, the DC component must be real
    si[...,0] = 0
    
    # Combine power + corrected phase to Fourier components
    s  = sr + 1J * si
    
    # Transform to real time series & scale to unit variance
    y = irfft(s, n=samples, axis=-1) / sigma
    
    return y
