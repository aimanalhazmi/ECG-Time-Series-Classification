import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import math


class STFTLayer(nn.Module):
    def __init__(self, n_fft, hop_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft
        self.window = torch.hann_window(self.win_length)

    def forward(self, x_padded, lengths_raw):
        self.window = self.window.to(x_padded.device)
        x_stft_complex = torch.stft(x_padded,
                                    n_fft=self.n_fft,
                                    hop_length=self.hop_length,
                                    win_length=self.win_length,
                                    window=self.window,
                                    return_complex=True,
                                    center=True)
        x_mag = x_stft_complex.abs()
        new_lengths = torch.div(lengths_raw.float(), self.hop_length, rounding_mode='floor') + 1
        return x_mag, new_lengths.long()


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size, pool_kernel_size, pool_stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel_size, padding='same')
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
        self.pool_kernel_time = pool_kernel_size[1]
        self.pool_stride_time = pool_stride[1]

    def forward(self, x, lengths):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        new_lengths = torch.floor((lengths.float() - self.pool_kernel_time) / self.pool_stride_time) + 1
        return x, new_lengths.long()


class ECGClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.stft = STFTLayer(cfg.N_FFT, cfg.HOP_LENGTH)

        current_freq_dim = cfg.N_FFT // 2 + 1

        self.conv1 = ConvBlock(1,
                               cfg.CONV1_OUT_CHANNELS,
                               cfg.CONV1_KERNEL_SIZE,
                               cfg.CONV1_POOL_KERNEL_SIZE,
                               cfg.CONV1_POOL_STRIDE)
        current_freq_dim = math.floor((current_freq_dim - cfg.CONV1_POOL_KERNEL_SIZE[0]) / cfg.CONV1_POOL_STRIDE[0]) + 1

        self.conv2 = ConvBlock(cfg.CONV1_OUT_CHANNELS,
                               cfg.CONV2_OUT_CHANNELS,
                               cfg.CONV2_KERNEL_SIZE,
                               cfg.CONV2_POOL_KERNEL_SIZE,
                               cfg.CONV2_POOL_STRIDE)
        current_freq_dim = math.floor((current_freq_dim - cfg.CONV2_POOL_KERNEL_SIZE[0]) / cfg.CONV2_POOL_STRIDE[0]) + 1

        rnn_input_features = cfg.CONV2_OUT_CHANNELS * current_freq_dim

        self.rnn = nn.LSTM(input_size=rnn_input_features,
                           hidden_size=cfg.RNN_HIDDEN_SIZE,
                           num_layers=cfg.RNN_NUM_LAYERS,
                           batch_first=True)

        self.fc = nn.Linear(cfg.RNN_HIDDEN_SIZE, cfg.NUM_CLASSES)

    def forward(self, x, lengths):
        x_padded, original_lengths = pad_packed_sequence(x, batch_first=True)
        x_stft_mag, lengths_after_stft = self.stft(x_padded, original_lengths)
        x = torch.log2(x_stft_mag.clamp(min=1e-9)).unsqueeze(1)
        current_lengths = lengths_after_stft
        x, current_lengths = self.conv1(x, current_lengths)
        x, current_lengths = self.conv2(x, current_lengths)
        batch_size = x.size(0)
        x = x.view(batch_size, -1, x.size(3))
        x = x.permute(0, 2, 1)
        current_lengths = torch.clamp(current_lengths, min=1)
        x = pack_padded_sequence(x, current_lengths.cpu(),
                                 batch_first=True, enforce_sorted=False)
        _, (ht, ct) = self.rnn(x)
        x = self.fc(ht[-1])
        return x