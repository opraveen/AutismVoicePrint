import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io.wavfile as wav
from scipy.fftpack import fft, fftfreq
import scipy.signal as signal


def calculate_energy(aud_sample):
    '''
    :param aud_sample: Numpy 1D array representation of audio sample, sample size > 0
    :return: Mean energy of aud_sample, float
    '''

    energy = np.mean(aud_sample*aud_sample)
    return energy


def normalize_sample(aud_sample):
    '''
    :param aud_sample: Numpy 1D array representation of audio sample (int or float)
    :return: Numpy 1D array - normalized audio sample to [-1, 1]
    '''

    abs_max = np.max(np.abs(aud_sample))
    return aud_sample*1.0/abs_max


def is_periodic(aud_sample, sampling_rate = 8000):
    '''
    :param aud_sample: Numpy 1D array rep of audio sample
    :param sampling_rate: Used to focus on human speech freq range
    :return: True if periodic, False if aperiodic
    '''

    # TODO: Find a sensible threshold
    thresh = 1e-4

    # Use auto-correlation to find if there is enough periodicity in [50-400] Hz range
    values = signal.correlate(aud_sample, aud_sample)
    print(values.max, values.shape)

    # [50-400 Hz] corresponds to [2.5-20] ms OR [20-160] samples for 8 KHz sampling rate
    l_idx = int(sampling_rate*2.5/1000)
    r_idx = int(sampling_rate*20/1000)

    print(l_idx, r_idx)

    subset_values = values[l_idx:r_idx]

    print(subset_values.shape, np.argmax(subset_values), subset_values.max())

    if subset_values.max() < thresh:
        return False
    else:
        return True


sampling_rate = 8000
window_size = sampling_rate*50/1000  # 400 samples, equivalent to 50 ms
window_stride = sampling_rate*10/1000   # 80 samples, equivalent to 10 ms

energy_threshold = 1e-4  # TODO: For this to be useful, normalize the speech samples before calculating energy

(rate, sig) = wav.read("./audacity_samples/op.wav")

# Step 0: Pre-process the speech sample
# a. Down-sample to 8 MHz (should be enough for Autism detection - only human speech)
# b. Normalization [Apply gain s.t the sample data is in the range [-1.0, 1.0]
# c. Noise Cancellation (TODO)

# Somehow, the down-sampling results in amplitude rescaling - Why?
sig = signal.resample(sig, len(sig)*sampling_rate/rate)

sig = normalize_sample(sig)

# plt.plot(range(len(sig)), sig)
# plt.show()
# exit()

# For each window:
# 1. Filter out low energy samples
# 2. Filter out aperiodic data (ONLY periodic speech samples can be nasalized - References?)
# 3. Classifier to detect nasality index using FFT values as features. 'Phase' info can be included later
# 4. Measure nasality only using the windows reaching #3

for i in range(0, len(sig), window_stride):
# for i in range(8000, 9200, window_stride):

    window = sig[i:i+window_size]

    window_energy = calculate_energy(window)
    print(len(window), window.shape, window_energy)

    # Energy filter
    if window_energy < energy_threshold:
        continue

    # Periodicity check
    if is_periodic(window) is False:
        continue

    # FFT to shift to frequency domain - use frequency spectrum as features for classifier
    fft_values = abs(fft(window))
    print(sig.shape, window.shape, fft_values.shape)
    fft_freq = fftfreq(window.size, 1)
    print(len(fft_freq), fft_freq.shape)

    plt.plot(fft_freq, 20*scipy.log10(fft_values), 'x')
    plt.show()
    plt.plot(range(len(window)), window)
    plt.show()
    exit()
