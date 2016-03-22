import re
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io.wavfile as wav
from scipy.fftpack import fft, fftfreq
import scipy.signal as signal
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import linear_model
from sklearn.cluster import KMeans

def noise_removal(aud_sample):
    if (min(abs(aud_sample)) == 0):
      return aud_sample

    data = abs(np.copy(aud_sample))
    clf = KMeans(n_clusters = 2,n_init = 5)
    data = data.reshape(-1,1)
    clf.fit(data)
    if clf.cluster_centers_[0] < clf.cluster_centers_[1]:
      noise = 0
    else:
      noise = 1

    aud = np.copy(aud_sample)
    
    window = 500
    windowStride = 50
    for i in range(0,len(clf.labels_),windowStride):
        if sum(clf.labels_[i:i+window] == noise) == window:
            aud[i:i+window] = 0

    return aud

def classify_using_logistic(feat1, feat2, classifier):

    n_plus = len(feat1)
    n_minus = len(feat2)

    X = np.concatenate((feat1, feat2), axis=0)
    y = np.concatenate((np.zeros(n_plus), np.ones(n_minus)), axis=0)
    y = y + 1

    print(X.shape, y.shape, n_plus, n_minus, feat1.shape, feat2.shape)

    print("Score using logistic regression on training data is ", classifier.score(X, y))


def train_using_logistic(feat1, feat2):

    n_plus = len(feat1)
    n_minus = len(feat2)

    X = np.concatenate((feat1, feat2), axis=0)
    y = np.concatenate((np.zeros(n_plus), np.ones(n_minus)), axis=0)
    y = y + 1

    print(X.shape, y.shape, n_plus, n_minus, feat1.shape, feat2.shape)

    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(X, y)

    print("Score using logistic regression on training data is ", logreg.score(X, y))
    return logreg

def preprocess_sample(aud_sample,rate):
    # Step 0: Pre-process the speech sample
    # a. Down-sample to 8 MHz (should be enough for Autism detection - only human speech)
    # b. Normalization [Apply gain s.t the sample data is in the range [-1.0, 1.0]
    # c. Noise Cancellation

    proc_sample = signal.resample(aud_sample, len(aud_sample)*SAMPLING_RATE/rate)

    if np.max(proc_sample) > 1.0:
        proc_sample = proc_sample*1.0/pow(2, 15)

    proc_sample = noise_removal(proc_sample)
    return proc_sample

def is_periodic(aud_sample, SAMPLING_RATE = 8000):
    '''
    :param aud_sample: Numpy 1D array rep of audio sample
    :param SAMPLING_RATE: Used to focus on human speech freq range
    :return: True if periodic, False if aperiodic
    '''

    threshold = 20

    # Use auto-correlation to find if there is enough periodicity in [50-400] Hz range
    values = signal.correlate(aud_sample, aud_sample, mode='full')

    # [50-400 Hz] corresponds to [2.5-20] ms OR [20-160] samples for 8 KHz sampling rate
    l_idx = int(SAMPLING_RATE*2.5/1000)
    r_idx = int(SAMPLING_RATE*20/1000)
    values = values[len(values)/2:]

    subset_values = values[l_idx:r_idx]

    if np.argmax(subset_values) < threshold:
        return False
    else:
        return True


def create_labeled_data(aud_sample, nasal=0):

    num_windows = (len(aud_sample) - WINDOW_SIZE)/WINDOW_STRIDE

    features = np.zeros((num_windows, WINDOW_SIZE))
    labels = np.zeros(num_windows)

    idx = 0
    for i in range(0, len(aud_sample), WINDOW_STRIDE):

        window = aud_sample[i:i+WINDOW_SIZE]
        for j in range(len(window), WINDOW_SIZE):
            window = np.append(window,0)

        if is_periodic(window) is False:
           continue

        # FFT to shift to frequency domain - use frequency spectrum as features
        fft_values = abs(fft(window))

        feat = 20*scipy.log10(fft_values)

        features[idx:, ] = feat
        labels[idx] = nasal
        idx += 1
    return features[0:idx, ], labels[0:idx]

def prepareData(path):
    normal_files = os.listdir(path + "/Normal/")
    nasal_files = os.listdir(path + "/Nasalized/")
    normal_features = np.zeros((1,400))
    normal_labels = np.zeros((1,1))
    nasal_features = np.zeros((1,400))
    nasal_labels = np.zeros((1,1))

    for filename in normal_files:
        (rate, sig) = wav.read(path + "/Normal/" + filename)
        sig = sig[:,0]
        sig = preprocess_sample(sig,rate)
        features, labels = create_labeled_data(sig, nasal=0)
        normal_features = np.append(normal_features,features,axis = 0)
    for filename in nasal_files:
        (rate, sig) = wav.read(path + "/Nasalized/" + filename)
        sig = sig[:,0]
        sig = preprocess_sample(sig,rate)
        features, labels = create_labeled_data(sig, nasal=1)
        nasal_features = np.append(nasal_features,features,axis = 0)
    normal_features = normal_features[1:]
    nasal_features = nasal_features[1:]
    return (normal_features,nasal_features)



SAMPLING_RATE = 8000
WINDOW_SIZE = SAMPLING_RATE*50/1000
WINDOW_STRIDE = SAMPLING_RATE*10/1000

normal_features, nasal_features = prepareData("data")

normal_len = (normal_features.shape[0] / 10 ) * 8
nasal_len = (nasal_features.shape[0] / 10 ) * 8

normal_features_train = normal_features[0:normal_len,:]
nasal_features_train = nasal_features[0:nasal_len,:]
normal_features_test = normal_features[normal_len:,:]
nasal_features_test = nasal_features[nasal_len:,:]

classifier = train_using_logistic(normal_features_train, nasal_features_train)

classify_using_logistic(normal_features_test, nasal_features_test, classifier)