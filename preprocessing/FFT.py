import torch


class FFT:
    def __init__(self, input_sig, fs=60.0):
        self.input_sig = input_sig
        self.fs = fs

    def fft(self, dc_removal=True, plot=False):
        if dc_removal:
            pass
            # self.input_sig = Cleansing(self.input_sig, cpu=True).dc_removal()
        amp = torch.fft.rfft(self.input_sig, dim=-1)
        freq = torch.fft.rfftfreq(self.input_sig.shape[-1], 1 / self.fs)
        if plot:
            import matplotlib.pyplot as plt

            plt.title("FFT")
            plt.plot(freq, amp)
            plt.grid(True)
            plt.show()
        return amp, freq

    def peak_freq(self):
        amp, freq = self.fft(dc_removal=True)
        return freq[torch.argmax(amp)]
