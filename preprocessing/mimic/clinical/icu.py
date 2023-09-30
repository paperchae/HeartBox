import pandas as pd
import numpy as np

from utils.config import get_config
from utils.pandas_utils import *

class ICU:
    def __init__(self, icu_path: str, subject_list: list):
        self.root_path = icu_path + 'icu/'
        self.subject_list = subject_list

    def read_chartevents(self):
        chunk_size = 10 ** 6
        cnt = 0
        gender_list = []
        age_list = []
        disease_list = []

        for chunk in pd.read_csv(self.root_path + 'chartevents.csv.gz', chunksize=chunk_size, compression='gzip',
                                 header=0, sep=',', low_memory=False):
            chunk = loc(chunk, 'SUBJECT_ID', 'in', self.subject_list)
            print('test')
        pass

    def read_d_items(self):
        pass

    def read_icustays(self):
        pass

    def read_inputevents(self):
        pass

    def read_outputevents(self):
        pass

    def return_icu(self):
        pass



if __name__ == '__main__':
    cfg = get_config('../../../config.yaml')
    root_path = cfg.preprocess.root_path
    sub_path = cfg.preprocess.mimic.clinical_path
    path = root_path + sub_path

    icu = ICU(path)

