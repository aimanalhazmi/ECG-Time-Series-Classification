import numpy as np
from scipy.signal import resample


class ECGTransform:
    def __init__(
        self, cfg):
        self.do_add_noise = cfg.DO_ADD_NOISE
        self.do_amplitude_scaling = cfg.DO_AMPLITUDE_SCALING
        self.do_time_shift = cfg.DO_TIME_SHIFT
        self.do_frequency_augment = cfg.DO_FREQUENCY_AUGMENT
        self.shift_range = cfg.SHIFT_RANGE
        self.scale_range = cfg.SCALE_RANGE
        self.noise_level = cfg.NOISE_LEVEL
        self.freq_noise_level = cfg.FREQ_NOISE_LEVEL

    def __call__(self, signal):
        signal = np.asarray(signal, dtype=np.float32)
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

    def amplitude_scaling(self, signal):
        scale = np.random.uniform(*self.scale_range)
        return signal * scale

    def add_noise(self, signal):
        return signal + np.random.normal(0, self.noise_level, len(signal))

    def time_shift(self, signal, shift):
        return np.roll(signal, shift)

    def frequency_augment(self, signal):
        fft = np.fft.fft(signal)
        noise = np.random.normal(0, self.freq_noise_level, fft.shape)
        return np.fft.ifft(fft + noise)
