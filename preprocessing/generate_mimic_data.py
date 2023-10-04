import os
from utils.config import get_config
from tqdm import tqdm
import pandas as pd
import wfdb
import matplotlib.pyplot as plt
import numpy as np

from utils.wfdb_funcs import get_ecg_idx
from utils.pandas_utils import *
from mimic.ecg import ECG
from mimic.clinical.icu import ICU
from mimic.clinical.hosp import HOSP


class Clinical:
    """
    subject list를 받아서 clinical data를 읽어오는 class
    """

    def __init__(self, subject_list):
        cfg = get_config('../config.yaml')
        clinical_path = cfg.preprocess.root_path + cfg.preprocess.mimic.clinical_path
        icu_path = clinical_path + 'icu/'
        hosp_path = clinical_path + 'hosp/'
        self.hosp = HOSP(hosp_path, subject_list)
        self.icu = ICU(icu_path, subject_list)
        self.test = 'test'

    def merge_icu_hosp(self):
        temp = self.icu.read_chartevents()


class Patient:
    """
    subject_id 를 받아서 subject_id 에 해당하는 정보를 읽어오는 class
    """

    def __init__(self, subject_id):
        self.id = subject_id

    def get_subject_info(self, machine_df):
        """
        Read dataframe & get subject information.
        If there are many studies, return all studies.
        :return:
        """
        # study_id = study_per_patient[self.id]
        # study_df = machine_df.loc[machine_df['subject_id'] == self.id]
        subject_df = machine_df.loc[machine_df['subject_id'] == self.id]
        return subject_df

    def get_waveform_path(self, note_df):
        """
        get patient waveform_path according to self.id
        :param note_df:
        :return:
        """

        waveform_path = note_df.loc[note_df['subject_id'] == self.id]
        return waveform_path


if __name__ == '__main__':
    cfg = get_config('../config.yaml')
    root_path = cfg.preprocess.root_path
    sub_path = cfg.preprocess.mimic.ecg_path
    root_path += sub_path
    ecg = ECG(root_path)
    groups, patients_per_group, studies_per_patient = ecg.get_directory_structure()
    measurement_df = ecg.get_machine_measurement_info()
    machine_df = ecg.machine_df
    note_df = ecg.note_df

    study_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for s in study_list:
        target_patients = ecg.get_patient_over_n_study(studies_per_patient, study_num=s)
        # save target patients to csv
        target_patients_df = pd.DataFrame(target_patients)
        target_patients_df.to_csv(root_path + 'target_patients_{}.csv'.format(s))
        print('TEST')
    target_patients = ecg.get_patient_over_n_study(studies_per_patient, study_num=10)
    clinical = Clinical(target_patients)
    clinical.merge_icu_hosp()

    for t in target_patients:
        id = t
        p = Patient(id)
        subject_df = p.get_subject_info(machine_df)
        waveform_df = p.get_waveform_path(note_df)
        # total_df 가 안찍히는 경우는 waveform_df 가 비어있는 경우
        total_df = pd.merge(subject_df, waveform_df, on=['subject_id', 'study_id'])
        # TODO: ecg_time 이랑 note_seq 랑 순서가 안맞음, ecg_time 을 clinical database 에서 anchor year 들고와야함
        # TODO: 날짜를 x축, total_df의 report_n 을 y축으로 그래프 그리기
        # TODO: ecg_time, charttime 같은거 같음 하나 삭제 or 서로 다른거 삭제?
        different_df = total_df[total_df['ecg_time'] == total_df['charttime']]
        sorted_by_note_seq = total_df.sort_values(by=['note_seq'])
        waveforms = sorted_by_note_seq['waveform_path'].tolist()
        for idx, study in sorted_by_note_seq.iterrows():
            reports = study.filter(regex='report_')
            nan_deleted_reports = reports.to_numpy()[~pd.isna(reports.tolist())]
            plt.title('study_id:{} \nreport: {}'.format(study['study_id'], str(nan_deleted_reports)))
            waveform_path = root_path + study['waveform_path']
            lead_i = np.squeeze(wfdb.rdrecord(waveform_path, channels=[0]).p_signal)
            plt.plot(lead_i)
            plt.show()
            print('test')

    print('test')

    print(len(target_patients))
