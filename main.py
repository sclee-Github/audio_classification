import sklearn
import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display

import IPython.display as ipd

import warnings
warnings.filterwarnings('ignore')

PROJECT_HOME = './'
DATASET_DIR = PROJECT_HOME + 'dataset/DS1/ExperimentallyCollected/Bebop2_Test_File2.wav'
FIG_SIZE = (15,5)


if __name__ == '__main__':
    
    signal, sr = librosa.load(DATASET_DIR)

    # region Plot
    # (file_dir, file_id) = os.path.split(DATASET_DIR)
    # time = np.linspace(0, len(x)/sr, len(x))

    # fig, ax1 = plt.subplots() # plot
    # ax1.plot(time, x, color = 'b', label='speech waveform')
    # ax1.set_ylabel("Amplitude") # y 축
    # ax1.set_xlabel("Time [s]") # x 축
    # plt.title(file_id) # 제목
    # plt.savefig(file_id+'.png')
    # plt.show()
    # endregion

    # Playing Audio        
    ipd.Audio(data=signal, rate=sr)

    # Visualizing Audio (waveform)
    plt.figure(figsize=FIG_SIZE)
    librosa.display.waveshow(signal, sr=sr)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")

    # region Visualizing Audio (spectrogram)
    # X = librosa.stft(signal)
    # Xdb = librosa.amplitude_to_db(abs(X))
    # plt.figure(figsize=(11, 5))
    # librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    # plt.colorbar()
    # endregion

    # Visualizing Audio (power spectrum)    
    fft = np.fft.fft(signal)
    magnitude = np.abs(fft)

    freq = np.linspace(0,sr,len(magnitude))

    left_spectrum = magnitude[:int(len(magnitude)/2)]
    left_freq = freq[:int(len(magnitude)/2)]

    mod_left_freq = left_freq[left_freq<8001]
    mod_left_spectrum = left_spectrum[:len(mod_left_freq)]

    plt.figure(figsize=FIG_SIZE)
    plt.plot(mod_left_freq, mod_left_spectrum)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (mag./Hz)")
    plt.title("Power spectrum")
    plt.show()

    # Feature Extraction (MFCC)
    # n_fft = 2048 # number of sample per each frame
    # hop_length = 512 # total number of frame

    S = librosa.feature.melspectrogram(signal, sr=sr, n_mels=128, fmax=8000)
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S), sr=sr, n_mfcc=40)

    # mfccs = librosa.feature.mfcc(signal, sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=40)
    # mfccs = librosa.feature.mfcc(signal, sr, n_mfcc=40)
    print(mfccs.shape)

    # Visualizing Audio (MFCC)   
    plt.figure(figsize=FIG_SIZE)
    # librosa.display.specshow(mfccs, sr=sr, hop_length=hop_length, x_axis='time')
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.xlabel("Time")
    plt.ylabel("MFCC coefficients")
    plt.colorbar()
    plt.title("MFCCs")
    # plt.plot(signal[1000:5000])
    # plt.tight_layout()
    plt.show()    

    # Feature Scaling (Mean Normalization)
    mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
    print(mfccs.mean(axis=1)) # zero mean of each coefficient dimension
    print(mfccs.var(axis=1))  # unit variance of each coefficient dimension

    plt.figure(figsize=FIG_SIZE)
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.xlabel("Time")
    plt.ylabel("MFCC coefficients")
    plt.colorbar()
    plt.title("MFCCs")
    # plt.tight_layout()
    plt.show()
    





    
    
    
