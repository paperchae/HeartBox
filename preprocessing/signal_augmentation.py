from Handler import Manipulation, Cleansing


class Augmentation:
    def __init__(self, input_sig, input_fs, target_fs, cfg):
        self.input_sig = input_sig
        self.input_fs = input_fs
        self.target_fs = target_fs
        self.cfg = cfg

    def augment1(self):
        """
        Down-sample and normalize

        :return:
        """
        aug1 = Manipulation(self.input_sig).down_sample(from_fs=self.input_fs, to_fs=self.target_fs)
        aug1 = Manipulation(aug1).normalize(self.cfg.option.normalize)

        return aug1

    def augment2(self):
        """
        Detrend, down-sample and normalize

        :return:
        """
        aug2 = Cleansing(self.input_sig, fs=self.input_fs, cpu=True).detrend(mode=self.cfg.option.detrend)
        aug2 = Manipulation(aug2).down_sample(from_fs=self.input_fs, to_fs=self.target_fs)
        aug2 = Manipulation(aug2).normalize(self.cfg.option.normalize)

        return aug2

    def augment3(self):
        """
        Denoise, down-sample and normalize

        :return:
        """
        aug3 = Cleansing(self.input_sig).noise_removal(self.cfg.option.denoise)
        aug3 = Manipulation(aug3).down_sample(from_fs=self.input_fs, to_fs=self.target_fs)
        aug3 = Manipulation(aug3).normalize(self.cfg.option.normalize)

        return aug3

    def augment4(self):
        """
        # TODO: Verification needed through visualization
        Flip baseline of original signal, down-sample and norma lize

        :return:
        """
        aug4 = Cleansing(self.input_sig).baseline_correction(fs=self.input_fs, flip_baseline=True)
        aug4 = Manipulation(aug4).down_sample(from_fs=self.input_fs, to_fs=self.target_fs)
        aug4 = Manipulation(aug4).normalize(self.cfg.option.normalize)

        return aug4
