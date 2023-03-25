"""Code to calculate speech envelopes. Credit: Jonas Auernheimer"""
import numpy as np

import librosa as lb
import pyNSL
import mne
from scipy.signal import resample_poly, butter

def get_envelope_modes(track, sr=16000, fds=1000, filt_bank = 'CF_8kHz', frame_len=1, filt=[75, 150], f_thr=300, lowpass=True, 
                       lowp_freq = 120,  **filt_kwargs):
    
    '''
    envelope modulation feature:
    
    - frame_len: frame length of auditory spectrogram (e.g. 1ms -> 1kHz sampling rate)
    - filt_bank: filter bank audio signal from aud24.mat
    
    EDIT: IIR filter used
    '''
    
    CFs = pyNSL.pyNSL.get_filterbank()[filt_bank].squeeze()[:-1]
    pick_CFs = np.where(CFs > f_thr)[0]
    
    aud_spec = pyNSL.pyNSL.wav2aud(track, sr, [frame_len, 8, -2, -1])
    freq_mod = []
    
    for i in pick_CFs:
        freq_mod.append(mne.filter.filter_data(aud_spec[:,i], sfreq=fds, l_freq=filt[0], h_freq=filt[1],
                                               verbose=False, **filt_kwargs))
        
    if lowpass:
        freq_lowp = []
        for band in freq_mod:
            freq_lowp.append(mne.filter.filter_data(band, sfreq=fds, l_freq=None, h_freq=lowp_freq, verbose=False))
        env_mod = np.array(freq_lowp).mean(0)
        
    else:
        env_mod = np.array(freq_mod).mean(0)
    
    return env_mod



def calculate_env_modulations(audio_path):
    """Calculates gammatone envelope of a raw speech file.

     -- the calculation of the envelope is based on the
     following publication:
     "Auditory-Inspired Speech Envelope Extraction Methods for Improved EEG-Based
      Auditory Attention Detection in a Cocktail Party Scenario" (Biesmans et al., 2016)

     parameters
     ----------
     audio_path: str
        Audio file path
     power_factor: float
        Power used in power law relation, which is used to model
        relation between perceived loudness and actual speech
        intensity
     target_fs: int
        Sampling frequency of the calculated envelope

    returns
    -------
    numpy.ndarray
        Envelope of a speech
    """
    speech = np.load(audio_path)
    audio, fs = speech["audio"], speech["fs"]
    del speech

    assert int(fs) == 48000

    ds = 16000
    resampled = lb.resample(audio, fs, ds)

    filt_orig = butter(8, (75, 150), btype='bandpass', output='sos', fs=fs) # high gamma
    filt_high = 4000
    filt_low = 300

    modes = np.arange(50) 
    wdw = 75 # uniform window between upper and lower filter bound
    bin_gap = 80
    filt_range = []

    for mode in modes:
        filt_range.append((filt_low, filt_low+wdw)+mode*bin_gap)

    modes = get_envelope_modes(resampled, fds=500, frame_len=2)
    modes = resample_poly(modes, 128, 125)

    return modes[:, None]