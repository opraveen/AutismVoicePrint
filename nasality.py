import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io.wavfile as wav
from scipy.fftpack import fft, fftfreq
import scipy.signal as signal
from sklearn.decomposition import PCA
# from sklearn.lda import LDA  # deprecated?
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import linear_model


def classify_using_pca(feat1, feat2, num_comp=2):
    pca = PCA(n_components=num_comp)
    pca.fit(feat1)
    X = pca.transform(feat1)

    pca.fit(feat2)
    Y = pca.transform(feat2)

    plt.plot(X[:, 0], X[:, 1], 'ro')
    plt.plot(Y[:, 0], Y[:, 1], 'g+')
    plt.show()

    return X, Y


def classify_using_lda(feat1, feat2, num_comp=2):

    n_plus = len(feat1)
    n_minus = len(feat2)

    X = np.concatenate((feat1, feat2), axis=0)
    y = np.concatenate((np.zeros(n_plus), np.ones(n_minus)), axis=0)
    y += 1

    print(X.shape, y.shape, n_plus, n_minus, feat1.shape, feat2.shape)

    lda = LDA(n_components=num_comp)
    lda.fit(X, y)

    # TODO FIXME Why is this returning n_samples x 1, and not n_samples x 2?
    # Is it able to to differentiate using just 1 component? Crazy!!
    X_tr = lda.transform(X)

    print(X_tr.shape, lda.score(X, y))

    # CRAZY, we don't actually have the 2nd component from LDA
    X1 = np.concatenate((X_tr[0:n_plus], np.zeros((n_plus, 1))), axis=1)
    X2 = np.concatenate((X_tr[-n_minus:], np.ones((n_minus, 1))), axis=1)

    plt.plot(X1[:, 0], X1[:, 1], 'ro')
    plt.plot(X2[:, 0], X2[:, 1], 'g+')

    plt.ylim(-1, 3)
    plt.show()


def classify_using_logistic(feat1, feat2):

    n_plus = len(feat1)
    n_minus = len(feat2)

    X = np.concatenate((feat1, feat2), axis=0)
    y = np.concatenate((np.zeros(n_plus), np.ones(n_minus)), axis=0)
    y = y + 1

    print(X.shape, y.shape, n_plus, n_minus, feat1.shape, feat2.shape)

    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(X, y)

    print("Score using logistic regression on training data is ", logreg.score(X, y))


def normalize_sample(aud_sample):
    '''
    :param aud_sample: Numpy 1D array representation of audio sample (int or float)
    :return: Numpy 1D array - normalized audio sample to [-1, 1]
    '''

    abs_max = np.max(np.abs(aud_sample))
    return aud_sample*1.0/abs_max


def preprocess_sample(aud_sample):
    # Step 0: Pre-process the speech sample
    # a. Down-sample to 8 MHz (should be enough for Autism detection - only human speech)
    # b. Normalization [Apply gain s.t the sample data is in the range [-1.0, 1.0]
    # c. Noise Cancellation (TODO)

    # Somehow, the down-sampling results in amplitude rescaling - Why?
    proc_sample = signal.resample(aud_sample, len(aud_sample)*SAMPLING_RATE/rate)

    ## TODO Not recommended to normalize based on amplitude
    # proc_sample = normalize_sample(proc_sample)
    # Instead, convert from 16-bit PCM to float
    if np.max(proc_sample) > 1.0:
        proc_sample = proc_sample*1.0/pow(2, 15)

    # plt.plot(range(len(proc_sample)), proc_sample)
    # plt.show()
    # exit()

    return proc_sample


def calculate_energy(aud_sample):
    '''
    :param aud_sample: Numpy 1D array representation of audio sample, sample size > 0
    :return: Mean energy of aud_sample, float
    '''

    energy = np.mean(aud_sample*aud_sample)
    return energy


def is_periodic(aud_sample, SAMPLING_RATE = 8000):
    '''
    :param aud_sample: Numpy 1D array rep of audio sample
    :param SAMPLING_RATE: Used to focus on human speech freq range
    :return: True if periodic, False if aperiodic
    '''

    # TODO: Find a sensible threshold
    thresh = 1e-4

    # Use auto-correlation to find if there is enough periodicity in [50-400] Hz range
    values = signal.correlate(aud_sample, aud_sample)
    # print(values.max, values.shape)

    # [50-400 Hz] corresponds to [2.5-20] ms OR [20-160] samples for 8 KHz sampling rate
    l_idx = int(SAMPLING_RATE*2.5/1000)
    r_idx = int(SAMPLING_RATE*20/1000)
    # print(l_idx, r_idx)

    subset_values = values[l_idx:r_idx]

    # print(subset_values.shape, np.argmax(subset_values), subset_values.max())

    if subset_values.max() < thresh:
        return False
    else:
        return True


def create_labeled_data(aud_sample, nasal=0):
    # For each window:
    # 1. Filter out low energy samples
    # 2. Filter out aperiodic data (ONLY periodic speech samples can be nasalized - References?)
    # 3. Classifier to detect nasality index using FFT values as features. 'Phase' info can be included later
    # 4. Measure nasality only using the windows reaching #3

    num_windows = (len(aud_sample) - WINDOW_SIZE)/WINDOW_STRIDE

    features = np.zeros((num_windows, WINDOW_SIZE))
    labels = np.zeros(num_windows)

    idx = 0
    for i in range(0, len(aud_sample), WINDOW_STRIDE):

        window = aud_sample[i:i+WINDOW_SIZE]

        window_energy = calculate_energy(window)
        # print(len(window), window.shape, window_energy)

        # Energy filter
        if window_energy < energy_threshold:
            continue

        # Periodicity check
        if is_periodic(window) is False:
            continue

        # FFT to shift to frequency domain - use frequency spectrum as features
        fft_values = abs(fft(window))
        # print(aud_sample.shape, window.shape, fft_values.shape)

        feat = 20*scipy.log10(fft_values)
        # print(feat.shape, idx)

        features[idx:, ] = feat
        labels[idx] = nasal
        idx += 1

        # fft_freq = fftfreq(window.size, 1)
        # print(len(fft_freq), fft_freq.shape)
        # plt.plot(fft_freq, 20*scipy.log10(fft_values), 'x')
        # plt.show()
        # plt.plot(range(len(window)), window)
        # plt.show()
        # exit()

    return features[0:idx, ], labels[0:idx]


SAMPLING_RATE = 8000
WINDOW_SIZE = SAMPLING_RATE*50/1000  # 400 samples, equivalent to 50 ms
WINDOW_STRIDE = SAMPLING_RATE*10/1000   # 80 samples, equivalent to 10 ms

energy_threshold = 1e-4  # TODO: For this to be useful, normalize the speech samples before calculating energy

# (rate, sig) = wav.read("./audacity_samples/op.wav")
(rate, sig) = wav.read("./audacity_samples/sree_reg.wav")
(rate, nasal_sig) = wav.read("./audacity_samples/sree_nasal.wav")

print(sig.shape, nasal_sig.shape)

sig = preprocess_sample(sig)
nasal_sig = preprocess_sample(nasal_sig)

reg_features, reg_labels = create_labeled_data(sig, nasal=0)
nasal_features, nasal_labels = create_labeled_data(nasal_sig, nasal=1)

print(reg_features.shape, reg_features.mean())

# NOTE: PCA isn't helpful as the primary components of both nasal
# and non-nasal samples are likely to be similar
# Instead, try Fischer Linear Discriminant (supervised: Prof Saul's suggestion)
# classify_using_pca(reg_features, nasal_features, num_comp=2)


# LINEAR DISCRIMINANT ANALYSIS (Supervised) -
# Runs into the warning - "Variables are collinear", ends up generating only one component
# And the best part, That one-component alone seems to do a
# great job with classification (refer to LDA.png)
classify_using_lda(reg_features, nasal_features, num_comp=2)

# LOGISTIC REGRESSION: Gets to 100% accuracy with the initial samples
# classify_using_logistic(reg_features, nasal_features)

# TODO: Use histograms to visually interpret the differences b/w nasal and non-nasal samples


