from utils.config import get_config
from utils.pandas_utils import *


class HOSP:
    def __init__(self, hosp_path: str, subject_list: list):
        self.hosp_path = hosp_path
        self.subject_list = subject_list
        self.admissions = self.read_admissions()
        self.diagnoses_icd = self.read_diagnoses_icd()
        self.d_icd_diagnoses = self.read_d_icd_diagnoses()
        self.patients = self.patients()

    def read_admissions(self):
        admissions = pd.read_csv(self.hosp_path + 'admissions.csv.gz', compression='gzip', header=0, sep=',',
                                 low_memory=False)
        admissions = loc(admissions, 'subject_id', 'in', self.subject_list)
        return admissions
        pass

    def read_diagnoses_icd(self):
        diagnoses_icd = pd.read_csv(self.hosp_path + 'diagnoses_icd.csv.gz', compression='gzip', header=0, sep=',',
                                    low_memory=False)
        diagnoses_icd = loc(diagnoses_icd, 'subject_id', 'in', self.subject_list)
        return diagnoses_icd
        pass

    def read_d_icd_diagnoses(self):
        d_icd_diagnoses = pd.read_csv(self.hosp_path + 'd_icd_diagnoses.csv.gz', compression='gzip', header=0, sep=',',
                                      low_memory=False)
        d_icd_diagnoses = loc(d_icd_diagnoses, 'subject_id', 'in', self.subject_list)
        return d_icd_diagnoses
        pass

    def patients(self):
        patients = pd.read_csv(self.hosp_path + 'patients.csv.gz', compression='gzip', header=0, sep=',',
                               low_memory=False)
        patients = loc(patients, 'subject_id', 'in', self.subject_list)
        return patients
        pass
