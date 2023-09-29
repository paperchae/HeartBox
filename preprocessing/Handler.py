import torch
import numpy as np
from FFT import FFT
from skimage.restoration import denoise_wavelet


# TODO: implement denoise_wavelet with torch or numpy


class Cleansing:
    def __init__(self, signals, fs=125, cpu=False):
        self.signals = signals
        self.fs = fs
        self.cpu = cpu
        self.is_tensor = True if type(signals) == torch.Tensor else False
        if not self.is_tensor:
            self.signals = torch.from_numpy(self.signals.copy()).to(torch.float)
        else:
            self.signals = signals.to(torch.float)
        # if cpu:
        #     print('!Currently accessing Cleansing Class with cpu, it might take a long time.')
        #     pass
        # else:
        #     self.signals = self.signals.to('cuda')

    def bandpass_filter(self,
                        low_cut,
                        high_cut,
                        fs,
                        return_max=False):
        """
        Simple bandpass filter using RFFT

        :param low_cut: Low cut frequency
        :param high_cut: High cut frequency
        :param fs: Sampling rate of the signals
        :param return_max: Offset for baseline correction
        :return: filtered signals, offset(optional)
        """

        fft_result, frequencies = FFT(self.signals, fs=fs).fft()
        bpf = torch.logical_and(frequencies >= low_cut, frequencies <= high_cut)
        if not self.cpu:
            bpf = bpf.to('cuda')
        filtered_fft = fft_result * bpf
        filtered_signal = torch.fft.irfft(filtered_fft)
        if return_max:
            return filtered_signal, torch.max(torch.abs(filtered_signal), dim=-1, keepdim=True)[0]
        else:
            return filtered_signal

    def baseline_correction(self,
                            fs=125,
                            flip_baseline: bool = False):
        """
        Baseline correction with bandpass filter [0, 0.5] Hz

        :param fs: sampling rate for FFT used in bandpass filter
        :param flip_baseline: If True, flip the baseline to negative
        :return: Baseline corrected signals
        """
        baseline, offset = self.bandpass_filter(low_cut=0, high_cut=0.5, fs=fs, return_max=True)
        if flip_baseline:
            return self.signals - (baseline + offset) + (-baseline)
        else:
            return self.signals - (baseline + offset)
        # plt.plot(ecg_temp[0] - Cleansing(ecg_temp).bandpass_filter(0, 0.5, fs=125)[0].cpu().numpy());
        # plt.show()
        # pass

    def detrend(self,
                Lambda=100,
                mode: str = 'total'):
        """
        * Use total option Only available with cuda device*
        This function applies a detrending filter to the 1D signals with linear trend.
        Using diagonal matrix D, with torch batch matrix multiplication

        Based on the following article "An advanced detrending method with application
        to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.

        ** It may take 10 GB of GPU memory for signals with 18000 length

        :param mode: 'total' or 'baseline'
            'baseline' option is for detrending signals like ECG which has most of the data centered at 0
            'total' option is for detrending signals like BVP, ABP which has less centered data relatively
        :param Lambda: Smoothing parameter
        :return: Detrended signals
        """
        if not torch.cuda.is_available():
            raise Exception('Cuda device is not available')
        # if not self.is_tensor:
        #     self.signals = torch.from_numpy(self.signals.copy())
        # signals = self.signals.to(torch.float).to('cuda')
        if self.signals.dim() == 1:
            self.signals = self.signals.unsqueeze(0)
        test_n, length = self.signals.shape

        if mode == 'total':
            # lambda = 100
            H = torch.eye(length).to('cuda')
            ones = torch.diag(torch.ones(length - 2)).to('cuda')
            zeros_1 = torch.zeros((length - 2, 1)).to('cuda')
            zeros_2 = torch.zeros((length - 2, 2)).to('cuda')

            D = torch.cat((ones, zeros_2), dim=-1) + \
                torch.cat((zeros_1, -2 * ones, zeros_1), dim=-1) + \
                torch.cat((zeros_2, ones), dim=-1)

            detrended_signal = torch.bmm(self.signals.unsqueeze(1),
                                         (H - torch.linalg.inv(H + (Lambda ** 2) * torch.t(D) @ D)).expand(test_n, -1,
                                                                                                           -1)).squeeze()

            if detrended_signal.dim() == 1:
                offset = torch.mean(self.signals, dim=-1)
            else:
                offset = torch.mean(self.signals, dim=-1, keepdim=True)
            detrended_signal += offset
            if self.cpu:
                return detrended_signal.cpu()
            else:
                return detrended_signal
        elif mode == 'baseline':
            return self.baseline_correction(fs=self.fs)

    def dc_removal(self):
        """
        Removes the dc value from the signals
        :return: dc removed signals
        """
        return self.signals - torch.mean(self.signals, dim=-1, keepdim=True)

    def noise_removal(self, mode: str = 'soft'):
        """
        Removes the noise from the signals
        :return: noise removed signals
        """

        # TODO: check ecg dtype
        return np.array([denoise_wavelet(ecg,
                                         method='VisuShrink',
                                         mode=mode,
                                         wavelet_levels=3,
                                         wavelet='sym9') for ecg in self.signals])


class Manipulation:
    """
    Signal Manipulation class
    Supports to_chunks, down-Sample, normalize
    signals: torch.tensor(n, length)
    """

    def __init__(self, signals):
        self.signals = signals
        self.is_tensor = True if type(signals) == torch.Tensor else False
        if not self.is_tensor:
            if self.signals.ndim == 1:
                self.signals = np.expand_dims(self.signals, axis=0)
        self.data_type = type(signals)
        # self.device = signals.get_device() if type(signals) == torch.Tensor else None
        self.device = 'cuda' if type(signals) == torch.Tensor else 'cpu'
        self.n, self.length = self.signals.shape

    def to_chunks(self,
                  chunk_size: int):
        """
        Split signals into chunks
        * if remainder exists, drop the last chunk for splitting
        :param chunk_size:
        :return: return signals in shape (n, -1, chunk_size)
        """
        if self.length < chunk_size:
            raise ValueError('Divider(Chunk size) is larger than signal length')

        return self.signals[:, :self.length - self.length % chunk_size].reshape(self.n, -1, chunk_size)

    def down_sample(self,
                    from_fs: int,
                    to_fs: int):
        """
        Down-sample signals from from_fs to to_fs
        :param from_fs: int, original sampling rate
        :param to_fs: int, target sampling rate
        :return: down sampled signals in shape (n, -1)
        """
        # if self.signals.ndim == 1:
        #     self.signals = np.expand_dims(self.signals, axis=0)
        if self.data_type == np.ndarray:
            return np.array([x[0::from_fs // to_fs] for x in self.signals])
        elif self.data_type == torch.Tensor:
            return torch.stack([x[0::from_fs // to_fs] for x in self.signals])

    def normalize(self,
                  mode='minmax'):
        """
        Normalize 1D signals
        :param mode: str, 'minmax' or 'zscore'
                     if 'minmax': normalize to [0, 1]
                     if 'zscore': normalize to 0 mean and 1 std
        :return: normalized signals
        """
        if self.data_type != torch.Tensor:
            self.signals = torch.from_numpy(self.signals.copy())
        if mode == 'minmax':
            min_val = torch.min(self.signals, dim=-1, keepdim=True)[0]
            max_val = torch.max(self.signals, dim=-1, keepdim=True)[0]
            return (self.signals - min_val) / (max_val - min_val)
        elif mode == 'zscore':
            if self.signals.dtype != torch.float:
                self.signals = self.signals.to(torch.float)
            mean = torch.mean(self.signals, dim=-1, keepdim=True)
            std = torch.std(self.signals, dim=-1, keepdim=True)
            return (self.signals - mean) / std

    def trim_mask(self):
        """
        Trim unnecessary padding mask for signals with the longest signals for computing resource efficiency.
        :return: Trimmed mask Tensor
        """
        return self.signals[:, :torch.max(torch.sum(self.signals > 0, dim=-1))]

    def remove_negative(self):
        pass
