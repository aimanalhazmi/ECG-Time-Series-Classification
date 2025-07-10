import numpy as np
from scipy.signal import resample


class ECGTransform:
    def __init__(self, cfg):
        self.do_add_noise = cfg.DO_ADD_NOISE
        self.do_amplitude_scaling = cfg.DO_AMPLITUDE_SCALING
        self.do_time_shift = cfg.DO_TIME_SHIFT
        self.do_frequency_augment = cfg.DO_FREQUENCY_AUGMENT
        self.noise_level = cfg.NOISE_LEVEL
        self.scale_range = cfg.SCALE_RANGE
        self.shift_range = cfg.SHIFT_RANGE
        self.freq_noise_level = cfg.FREQ_NOISE_LEVEL

        self.do_time_stretching = cfg.DO_TIME_STRETCHING
        self.stretch_range = cfg.STRETCH_RANGE

        self.do_random_crop = cfg.DO_RANDOM_CROP
        self.crop_length = cfg.MIN_SEQ_LEN

    def __call__(self, signal):
        signal = np.asarray(signal, dtype=np.float32)
        if self.do_time_stretching:
            signal = self.time_stretching(signal)
        if self.do_random_crop:
            signal = self.random_cropping(signal)
        if self.do_amplitude_scaling:
            signal = self.amplitude_scaling(signal)
        if self.do_add_noise:
            signal = self.add_noise(signal)
        if self.do_time_shift:
            shift = np.random.randint(*self.shift_range)
            signal = self.time_shift(signal, shift)
        if self.do_frequency_augment:
            signal = self.frequency_augment(signal)

        return np.real(signal)

    def time_stretching(self, signal):
        """
        Stretches or compresses the signal in time.
        """
        stretch_factor = np.random.uniform(*self.stretch_range)
        new_len = int(len(signal) * stretch_factor)
        return resample(signal, new_len)

    def random_cropping(self, signal):
        """
        Crops the signal to a fixed length from a random starting point.
        Pads the signal if it's shorter than the crop length.
        """
        if len(signal) > self.crop_length:
            start = np.random.randint(0, len(signal) - self.crop_length)
            return signal[start:start + self.crop_length]
        elif len(signal) < self.crop_length:
            # Pad with zeros if signal is too short
            padding = self.crop_length - len(signal)
            return np.pad(signal, (0, padding), 'constant', constant_values=0)
        return signal

    def amplitude_scaling(self, signal):
        """
        Scales the amplitude of the signal by a random factor within a specified range.
        """
        scale = np.random.uniform(*self.scale_range)
        return signal * scale

    def add_noise(self, signal):
        """
        Adds Gaussian noise to the signal.
        """
        return signal + np.random.normal(0, self.noise_level, len(signal))

    def time_shift(self, signal, shift):
        """
        Shifts the signal in time by a specified number of samples.
        """
        return np.roll(signal, shift)

    def frequency_augment(self, signal):
        """
        Adds noise to the frequency domain representation of the signal.
        """
        fft = np.fft.fft(signal)
        noise = np.random.normal(0, self.freq_noise_level, fft.shape)
        return np.fft.ifft(fft + noise)