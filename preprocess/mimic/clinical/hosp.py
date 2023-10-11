from utils.pandas_utils import *
from preprocess.icd.icd_chapter import icd_9_chapters, icd_10_chapters


class HOSP:
    def __init__(self, hosp_path: str, subject_list: list, target_diagnoses: str = "circulatory"):
        """
        Read hospital data and return target dataframes,
        including admissions, diagnoses, patients.

        :param hosp_path: path of the hospital data
        :param subject_list: list of target subject id
        :param target_diagnoses: target diagnosis to read,

        """
        self.hosp_path = hosp_path
        self.subject_list = subject_list
        self.target_diagnoses = target_diagnoses

    def read_admissions(self):
        """
        Read admissions.csv.gz and return target subject_id dataframe

        :return admissions dataframe
        """

        print("Reading admissions.csv.gz...")

        admissions = pd.read_csv(self.hosp_path + 'admissions.csv.gz', compression='gzip', header=0, sep=',',
                                 low_memory=False)
        admissions = loc(admissions, 'subject_id', 'in', self.subject_list)
        admissions = admissions[["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "race"]]
        return admissions

    def read_d_icd_diagnoses(self, icd_code):
        """
        Read d_icd_diagnoses.csv.gz to get descriptions of icd code
        *** This function is not used in this class due to lack of chapter title***
        """

        print("Reading d_icd_diagnoses.csv.gz...")

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
