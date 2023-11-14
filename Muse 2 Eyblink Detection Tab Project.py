import pyautogui
from time import sleep
from pynput.keyboard import Key, Controller
from os import system as sys
from datetime import datetime
import numpy as np
from pylsl import StreamInlet, resolve_byprop, resolve_stream
import utils
from brainflow.data_filter import DataFilter, FilterTypes, NoiseTypes
from scipy import signal

streams = resolve_byprop('type', 'EEG', timeout=2)
if len(streams) == 0:
    raise RuntimeError('Can\'t find EEG stream.')
inlet = StreamInlet(streams[0], max_chunklen=12)

#all lengths are in n seconds
BUFFER_LENGTH = 2
EPOCH_LENGTH = 0.5
OVERLAP_LENGTH = 0.2
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH

# 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear
INDEX_CHANNEL_LEFT = [1]
INDEX_CHANNEL_RIGHT = [2]
INDEX_CHANNELS = [INDEX_CHANNEL_LEFT, INDEX_CHANNEL_RIGHT]
#sampling rate
info = inlet.info()
fs = int(info.nominal_srate())
print(fs)


def vectorize(df, fs, filtering=False):
    index = len(df)
    feature_vectors = []
    if filtering == True:
      DataFilter.perform_bandpass(df[:], fs, 18.0, 22.0, 4,FilterTypes.BESSEL.value, 0)
      DataFilter.remove_environmental_noise(df[:], fs, NoiseTypes.SIXTY.value)
    for y in range(0,index,fs):
      f, Pxx_den = signal.welch(df[y:y+fs], fs=fs, nfft=256)
    # Delta 1-4
      ind_delta, = np.where(f < 4)
      meanDelta = np.mean(Pxx_den[ind_delta], axis=0)
    # Theta 4-8
      ind_theta, = np.where((f >= 4) & (f <= 8))
      meanTheta = np.mean(Pxx_den[ind_theta], axis=0)
    # Alpha 8-12
      ind_alpha, = np.where((f >= 8) & (f <= 12))
      meanAlpha = np.mean(Pxx_den[ind_alpha], axis=0)
    # Beta 12-30
      ind_beta, = np.where((f >= 12) & (f < 30))
      meanBeta = np.mean(Pxx_den[ind_beta], axis=0)
    # Gamma 30-100+
      ind_Gamma, = np.where((f >= 30) & (f < 40))
      meanGamma = np.mean(Pxx_den[ind_Gamma], axis=0)
      feature_vectors.insert(y, [meanDelta, meanTheta, meanAlpha,   meanBeta, meanGamma])
    powers = np.log10(np.asarray(feature_vectors))
    powers = powers.reshape(5)
    return powers

NUMBER_OF_CYCLES = 150 #run loop for 10000 cycles
data_holder_right = np.zeros(NUMBER_OF_CYCLES+1)
data_holder_left = np.zeros(NUMBER_OF_CYCLES+1)
i=0

while (i<NUMBER_OF_CYCLES):
    i = i +1
    #raw EEG buffer
    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
    # Compute the number of epochs in "buffer_length"
    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) /
    SHIFT_LENGTH + 1))
    # bands will be ordered: [delta, theta, alpha, beta, gamma]
    band_buffer = np.zeros((n_win_test, 5))
    #list of buffers for iteration
    buffers = [[eeg_buffer, eeg_buffer], [band_buffer, band_buffer]]

    #for index in range(len(INDEX_CHANNELS)):
    for index in [0, 1]:
        eeg_data, timestamp = inlet.pull_chunk(timeout=1, max_samples=int(SHIFT_LENGTH * fs))
        # Only keep the channel we're interested in
        ch_data = np.array(eeg_data)[:, INDEX_CHANNELS[index]]
        # Update EEG buffer with the new data
        buffers[0][index] = utils.update_buffer(buffers[0][index], ch_data)
        """ 3.2 COMPUTE BAND POWERS """
        # Get newest samples from the buffer
        data_epoch = utils.get_last_data(buffers[0][int(index)][0],int(EPOCH_LENGTH * fs))
        # Compute band powers
        band_powers = vectorize(data_epoch.reshape(-1), fs, filtering=True)
        #print (np.asarray([band_powers])[0].shape)
        buffers[1][index] = utils.update_buffer(buffers[1][index], np.array([band_powers]))

    # Delta is index 0; bands will be ordered: [delta, theta, alpha, beta, gamma]
    Band_sel = 0
    print(buffers[1][1][0][-1],'   ',buffers[1][0][0][-1])
    print(buffers[1][1][0][-1][Band_sel])
    data_holder_right[i] = buffers[1][1][0][-1][Band_sel]
    data_holder_left[i] = buffers[1][0][0][-1][Band_sel]
    
    
    if buffers[1][1][0][-1][Band_sel] < -1 and buffers[1][0][0][-1][Band_sel] < -1:
      print("""
      tab
      """)
      pyautogui.hotkey('ctrl', 'tab')
      buffers[1][1][0][-1][Band_sel] = 0
      buffers[1][0][0][-1][Band_sel] = 0
##    elif buffers[1][0][0][-1][Band_sel] < -1.2:
##      print("""
##      left
##      """)
##      #pyautogui.hotkey('ctrl', 'shift', 'tab')
##      buffers[1][0][0][-1][Band_sel] = 0

print("Stopped running the loop. Now saving data")

# write code to save data from data_holder to a csv file
np.savetxt("data.csv", [data_holder_left, data_holder_right], delimiter=",")
#np.savetxt("data_left.csv", data_holder_left, delimiter=",")
print("Data saved to data.csv")
# plot using matplotlib or excel
