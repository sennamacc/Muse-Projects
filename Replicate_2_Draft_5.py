import numpy as np # used for arrays
import matplotlib.pyplot as plt # used for plotting / creating visualizations
from pylsl import StreamInlet, resolve_byprop  # used to receive EEG data
import utils
from pynput.keyboard import Key, Controller # used to monitor/control keyboard inputs
from datetime import datetime
import pygame # used to play audio files

class Band:
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3

""" EXPERIMENTAL PARAMETERS """
# Modify these to change aspects of the signal processing

# Length of the EEG data buffer (in seconds)
BUFFER_LENGTH = 5

# Length of epochs
# Epochs are segment of EEG data
EPOCH_LENGTH = 1

# Amount of overlap between two consecutive epochs (in seconds)
OVERLAP_LENGTH = 0.8

# Amount to 'shift' the start of each next consecutive epoch
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH

# Index of which channel(s) (electrodes) being used
# 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear
INDEX_CHANNEL = [0]

# Mp3 name
MUSIC_FILENAME = None

#Asking the user to select a song
user_choice = input("""
Please select a song:
1. All I Want For Christmas Is You - Mariah Carey
2. Canon in D Major - Johann Pachelbel
3. Gentle rain sounds
""")

if user_choice == '1':
    MUSIC_FILENAME = 'Mariah Carey - All I Want For Christmas Is You.mp3'
elif user_choice == '2':
    MUSIC_FILENAME = 'Johann Pachelbel - Canon in D Major.mp3'
elif user_choice == '3':
    MUSIC_FILENAME = 'Gentle Rain With Smoothed Brown Noise.mp3'

if MUSIC_FILENAME:
    print(f"Alright! Playing {MUSIC_FILENAME}")
else:
    print("Invalid selection. Please type '1', '2', or ‘3’.")

if __name__ == "__main__":

    """ 1. CONNECT TO EEG STREAM """

    # Search for active LSL streams
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')

    # Set active EEG stream to inlet and apply time correction
    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    eeg_time_correction = inlet.time_correction()

    # Get the stream info and description
    info = inlet.info()
    description = info.desc()

    # Get the sampling frequency
    # This is an important value that represents how many EEG data points are collected in a second.
    fs = int(info.nominal_srate())

    """ 2. INITIALIZE BUFFERS """

    # Initialize raw EEG data buffer
    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
    filter_state = None  # for use with the notch filter

    # Compute the number of epochs in "buffer_length"
    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) /
                              SHIFT_LENGTH + 1))

    # Initialize the band power buffer (for plotting)
    # bands will be ordered: [delta, theta, alpha, beta]
    band_buffer = np.zeros((n_win_test, 4))


""" 3. GET DATA """

print('Press Ctrl-C in the console to break the while loop.')

try:
        keyboard = Controller()
        count=0
        # Initialize music player
        pygame.mixer.init()
        pygame.mixer.music.load(MUSIC_FILENAME)
        pygame.mixer.music.play(loops=-1)
        pygame.mixer.music.pause()
        # The following loop acquires data, computes band powers, and calculates neurofeedback metrics based on those band powers
        while True:
        
            """ 3.1 ACQUIRE DATA """
            # Obtain EEG data from the LSL stream
            eeg_data, timestamp = inlet.pull_chunk(
                timeout=1, max_samples=int(SHIFT_LENGTH * fs))

            # Only keep the channel we're interested in
            ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]

            # Update EEG buffer with the new data
            eeg_buffer, filter_state = utils.update_buffer(
                eeg_buffer, ch_data, notch=True,
                filter_state=filter_state)

            """ 3.2 COMPUTE BAND POWERS """
            # Get newest samples from the buffer
            data_epoch = utils.get_last_data(eeg_buffer,
                                             EPOCH_LENGTH * fs)

            # Compute band powers
            band_powers = utils.compute_band_powers(data_epoch, fs)
            band_buffer, _ = utils.update_buffer(band_buffer,
                                                 np.asarray([band_powers]))
            # Compute the average band powers for all epochs in buffer
            # This helps to smooth out noise
            smooth_band_powers = np.mean(band_buffer, axis=0)

            # print('Delta: ', band_powers[Band.Delta], ' Theta: ', band_powers[Band.Theta],
            #       ' Alpha: ', band_powers[Band.Alpha], ' Beta: ', band_powers[Band.Beta])
	
            #Assigning the varible to the beta/t
            beta_metric = smooth_band_powers[Band.Beta] / smooth_band_powers[Band.Theta]
            #Printing it's value
            print('Beta Concentration: ', beta_metric)

 # if beta_metric < 0.25 and (datetime.now() - t).seconds > 5:
            if beta_metric < 0.7:
                count=count+1
            if count>5:
                print("------Great job, keep concentrating :)------")
                pygame.mixer.music.unpause()
                count=0

            if beta_metric > 0.6:
                print ("-----------Get back to work >:(-----------")
                pygame.mixer.music.pause()

except KeyboardInterrupt:
    pygame.mixer.music.stop()
    pygame.mixer.quit()
    print('Bye, see you soon!')
