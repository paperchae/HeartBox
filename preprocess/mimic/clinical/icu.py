import pandas as pd
import numpy as np

from utils.config import get_config
from utils.pandas_utils import *


class ICU:
    def __init__(self, icu_path: str, subject_list: list):
        self.icu_path = icu_path
        self.subject_list = subject_list

    def read_d_items(self, itemid_list):
        items = pd.read_csv(self.icu_path + 'd_items.csv.gz', compression='gzip', header=0, sep=',', low_memory=False)
        target_items = loc(items, 'itemid', 'in', itemid_list)
        return target_items
        # print('test')

    def read_chartevents(self):
        chunk_size = 10 ** 6
        cnt = 0
        gender_list = []
        age_list = []
        disease_list = []

        for chunk in pd.read_csv(self.icu_path + 'chartevents.csv.gz', chunksize=chunk_size, compression='gzip',
                                 header=0, sep=',', low_memory=False):
            chunk = loc(chunk, 'subject_id', 'in', self.subject_list)
            test = self.read_d_items(chunk['itemid'])
            in_events = self.read_inputevents()
            out_events = self.read_outputevents()
            print('test')

    def read_icustays(self):
        pass

    def read_inputevents(self):
        inputs = pd.read_csv(self.icu_path + 'inputevents.csv.gz', compression='gzip', header=0, sep=',', low_memory=False)
        return inputs
        # pass

    def read_outputevents(self):
        outputs = pd.read_csv(self.icu_path + 'outputevents.csv.gz', compression='gzip', header=0, sep=',', low_memory=False)
        return outputs
        # pass

    def return_icu(self):
        pass


if __name__ == '__main__':
    cfg = get_config('../../../config.yaml')
    root_path = cfg.preprocess.root_path
    sub_path = cfg.preprocess.mimic.clinical_path
    path = root_path + sub_path

    icu = ICU(path)
