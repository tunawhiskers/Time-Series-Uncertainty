"""
functions for generated simulated noise
"""

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