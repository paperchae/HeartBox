from utils.config import get_config
from utils.pandas_utils import *
from preprocess.icd import *


class HOSP:
    def __init__(self, hosp_path: str, subject_list: list, target_diagnoses: str):
        self.hosp_path = hosp_path
        self.subject_list = subject_list
        self.diagnoses = target_diagnoses

        self.admissions = self.read_admissions()
        self.diagnoses_icd = self.read_diagnoses_icd()
        self.d_icd_diagnoses = self.read_d_icd_diagnoses(self.diagnoses_icd['icd_code'].unique())
        self.patients = self.patients()

    def read_admissions(self):
        admissions = pd.read_csv(self.hosp_path + 'admissions.csv.gz', compression='gzip', header=0, sep=',',
                                 low_memory=False)
        admissions = loc(admissions, 'subject_id', 'in', self.subject_list)
        return admissions

    def read_diagnoses_icd(self):
        diagnoses_icd = pd.read_csv(self.hosp_path + 'diagnoses_icd.csv.gz', compression='gzip', header=0, sep=',',
                                    low_memory=False)
        diagnoses_icd = loc(diagnoses_icd, 'subject_id', 'in', self.subject_list)
        icd_9_patients = loc(diagnoses_icd, 'icd_version', 'eq', 9)
        icd_10_patients = loc(diagnoses_icd, 'icd_version', 'eq', 10)
        icd_9_code = self.read_d_icd_diagnoses(icd_9_patients['icd_code'].unique())
        icd_9_code_2 = loc(icd_9_code, 'icd_code', 'in', get_code_from_icd_dict(self.diagnoses))
        icd_10_code = self.read_d_icd_diagnoses(icd_10_patients['icd_code'].unique())
        return diagnoses_icd

    def read_d_icd_diagnoses(self, icd_code):
        d_icd_diagnoses = pd.read_csv(self.hosp_path + 'd_icd_diagnoses.csv.gz', compression='gzip', header=0, sep=',',
                                      low_memory=False)
        # get target icd code dataframe
        d_icd_diagnoses = loc(d_icd_diagnoses, 'icd_code', 'in', icd_code)
        # screen out the icd code that is not in the target icd code dataframe
        # d_icd_diagnoses = loc(d_icd_diagnoses, "icd_code", "in", get_code_from_icd_dict(self.diagnoses))
        return d_icd_diagnoses

    def patients(self):
        patients = pd.read_csv(self.hosp_path + 'patients.csv.gz', compression='gzip', header=0, sep=',',
                               low_memory=False)
        patients = loc(patients, 'subject_id', 'in', self.subject_list)
        return patients
