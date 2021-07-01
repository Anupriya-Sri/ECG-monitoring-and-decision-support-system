from ecgdetectors import Detectors
from hrv import HRV
import pywt
import numpy as np
import scipy


class Extractor:

    def __init__(self, fs=500):
        self.hrv_obj = HRV(fs)
        self.detector = Detectors(fs)
    
    #-------------------------call this function to get the features----------------------------
    def get_features(self, signal, neigh=8, level=3, wname='db1', fs=500, win_left=90, win_right=90):
        """
        Parameters:
        signal : shape is (length_of_signal), A single lead ecg signal
        fs : sampling frequency of the signal
        win_left : size of window to the left of r-peak
        win_right : size of window to the right of r-peak
        level : The level of wavelet transform
        wname : The name of wavelet transform to use
        
        Returns:
        All the featues as a tuple
        """

        r_peaks = self.get_r_peaks(signal)
        mean_rr, sd_rr, nn_50, pnn_50 = self.get_rr_features(r_peaks)
        mean_hr, sd_hr, max_hr, min_hr = self.get_hr_features(signal, r_peaks)

        beats = self.signal_to_beats(signal, win_left, win_right)
        wavelet_data = self.wavelet_transform(beats, level, wname)
        lbp_data = self.lbp_beats(beats)
        hos_data = self.HOS(beats, win_left=90, win_right=90, n_intervals=6)
        return mean_rr, sd_rr, nn_50, pnn_50, mean_hr, sd_hr, max_hr, min_hr, wavelet_data, lbp_data, hos_data

    def get_r_peaks(self, data):
        """
        Parameters:
        data : shape is (length_of_signal), A single lead ecg signal
        
        Returns:
        array : returns the 1-D coordinates of R peaks
        """
        
        r_peaks = self.detector.christov_detector(data)
        return r_peaks

    def get_rr_features(self, r_peaks):
        """
        Parameters:
        r_peaks : list of coordinates of RR preaks for each signal
        (number_of_rr_peaks)
        Returns:
        Mean and SD of RR interval, nn50 and pnn50
        """

        diff = np.abs(np.array(r_peaks[1:]) - np.array(r_peaks[:-1]))

        mean_rr = np.mean(diff)
        sd_rr = self.hrv_obj.SDNN(r_peaks)
        nn_50 = self.hrv_obj.NN50(r_peaks)
        pnn_50 = self.hrv_obj.pNN50(r_peaks)

        return mean_rr, sd_rr, nn_50, pnn_50



    def get_hr_features(self, data, r_peaks):
        """
        Parameters:
        data : shape is (length_of_signal), A single lead ecg signal
        r_peaks : list of coordinates of RR preaks for each signal
        (number_of_rr_peaks)
        
        Returns:
        Mean, SD, Minimum, Maximum of heart rate for each signal
        """

        bpm = self.hrv_obj.HR(r_peaks)
        mean_hr = np.mean(bpm)
        sd_hr = np.std(bpm)
        max_hr = np.max(bpm)
        min_hr = np.min(bpm)
        
        return mean_hr, sd_hr, max_hr, min_hr

    #-------------------------Beats extraction ----------------------------
    def signal_to_beats(self, signal, win_left, win_right):
        """
        Parameters:
        signal : shape is (length_of_signal), A single lead ecg signal
        win_left : size of window to the left of r-peak
        win_right : size of window to the right of r-peak
        
        Returns:
        beats : A 2D array. Each row is a beat of size win_left + win_right data  points with an r_peak at center
        """
        r_peaks = self.detector.christov_detector(signal)

        # beats - each row is beat of size win_left + win_right
        beats = np.empty(( len(r_peaks)-2, win_left + win_right), dtype=np.object) # -2 because we are leaving first nad lst r_peak it might be incomplete

        j=1 # Skipping the first r_peak
        for i in range(0, len(beats)):
            left = r_peaks[j] - win_left
            right = r_peaks[j] + win_right
            beats[i, :] = signal[left:right]
            j = j + 1
        
        return beats
    
    #-------------------------wavelet transformation ----------------------------
    def wavelet_transform(self, beats, level=3, wname='db1'):
        """
        Parameters:
        beats : It is the array of beats. One signal broken down into beats
        level : The level of wavelet transform
        wname : The name of wavelet transform to use

        Returns:
        The wavelet coefficients averaged for all beats
        """

        wavelet = pywt.Wavelet(wname)

        coeffs = pywt.wavedec(beats[0, :],  wavelet, level=level)
        n = coeffs[0].shape[0]
        m = beats.shape[0]

        # placeholder for wavelet data
        wavelet_data = np.empty((m,n), dtype=np.object)
        for i in range(0, m):
            coeffs = pywt.wavedec(beats[i, :], wavelet, level=level)
            wavelet_data[i,:] = coeffs[0]

        return np.mean(wavelet_data, axis=0)
    
    
    #-------------------------LBP single beat ----------------------------
    @staticmethod
    def compute_Uniform_LBP(beat, neighbors=8):
        """
        Parameters:
        beat : One beat of a signal, r_peak at the center
        
        Returns:
        Local binary pattern ids for one beat
        """
        
        uniform_pattern_list = np.array(
            [0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112, 120, 124, 126,
            127, 128,
            129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227, 231, 239, 240, 241, 243, 247, 248,
            249, 251, 252, 253, 254, 255]
        )


        hist_u_lbp = np.zeros(59)

        avg_win_size = 2
        inter = neighbors//2

        for i in  list(range(-inter, 0)) + list(range(1, (inter+1))):
            pattern = np.zeros(neighbors)
            ind = 0
            for n in list(range(-inter, 0)) + list(range(1, inter+ 1)):
                if beat[i] > beat[i+n]:
                        pattern[ind] = 1
                        ind += 1

            # Convert pattern to id-int 0-255 (for neighbors == 8)
            pattern_id = int("".join(str(c) for c in pattern.astype(int)), 2)

            # Convert id to uniform LBP id 0-57 (uniform LBP)  58: (non uniform LBP)
            if pattern_id in uniform_pattern_list:
                pattern_uniform_id = int(np.argwhere(uniform_pattern_list == pattern_id))
            else:
                pattern_uniform_id = 58 # Non uniforms patterns use

            hist_u_lbp[pattern_uniform_id] += 1.0

        return hist_u_lbp


    #-------------------------LBP full signal -------------------
    def lbp_beats(self, beats):
        """
        Parameters:
        beats : It is the array of beats. One signal broken down into beats
        
        Returns:
        lbp_data : LBP averaged over all beats
        """

        m = beats.shape[0]
        n = 59 # lbp gives 59 values

        # array to store LBP  data
        lbp_data = np.empty((m, n), dtype=np.object)
        for i in range(0, m):
            lbp_data[i,:] = self.compute_Uniform_LBP(beats[i,:], neighbors=8)

        return np.mean(lbp_data, axis=0)
    

    #-------------------------HOS ----------------------------
    def HOS(self, beats, win_left, win_right, n_intervals=6):
        """
        Parameters:
        beats : It is the array of beats. One signal broken down into beats
        
        Returns:
        hos_data : HOS values averaged over all beats
        """

        lag = round((win_left + win_right) // n_intervals)

        m, n = beats.shape
        #hos gives 10 features for one beat
        hos_data = np.empty((m, 10), dtype=np.object)
        for j in range(0,m): # For each beat
            hos_b = np.zeros(((n_intervals - 1) * 2))
            for i in range(0, n_intervals - 1):
                pose = (lag * (i + 1))
                interval = beats[j,:][(pose - (lag // 2)):(pose + (lag // 2))]
                # Skewness
                hos_b[i] = scipy.stats.skew(interval, 0, True)
                # Kurtosis
                hos_b[5 + i] = scipy.stats.kurtosis(interval, 0, False, True)

            hos_data[j,:]=hos_b

        return np.mean(hos_data, axis=0)






