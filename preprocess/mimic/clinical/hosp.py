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

    def read_diagnoses_icd(self):
        """
        Read diagnoses_icd.csv.gz and return target icd code dataframe

        1. Find the chapter title from the target diagnosis per icd version
        2. Read icd code dataframes & diagnoses_icd.csv.gz
        3. Filter out the subject_id that is not in the subject_list
        4. Find the target icd code from the icd code dataframe
        5. Merge the icd code dataframe with the diagnosis dataframe

        : return diagnoses dataframe with detailed descriptions
        """

        print("Reading diagnoses_icd.csv.gz...")

        def _retrieve_chapter_title(target_title):
            """
            Find the chapter title from the target diagnosis per icd version
            :param target_title: target diagnosis
            """
            version = [icd_9_chapters, icd_10_chapters]
            title = {"9": 0, "10": 0}
            version_cnt = 9
            for v in version:
                for chap, value in v.items():
                    if target_title.lower() in value["title"].lower():
                        title[str(version_cnt)] = value["title"]
                    else:
                        continue
                version_cnt += 1

            if all(list(title.values())):
                return title
            else:
                raise ValueError("Invalid chapter, check icd_chapter.py")

        # Find the chapter title from the target diagnosis per version
        chapter_titles = _retrieve_chapter_title(self.target_diagnoses)

        # Read icd code dataframes & diagnoses_icd.csv.gz
        icd_9 = pd.read_csv('../preprocess/icd/result/icd_9.csv',
                            header=0, sep=',', low_memory=False)
        icd_10 = pd.read_csv('../preprocess/icd/result/icd_10.csv',
                             header=0, sep=',', low_memory=False).drop(columns=['IS_HEADER'])
        diagnoses = pd.read_csv(self.hosp_path + 'diagnoses_icd.csv.gz',
                                compression='gzip', header=0, sep=',', low_memory=False)

        # Filter out the subject_id that is not in the subject_list
        diagnoses = loc(diagnoses, 'subject_id', 'in', self.subject_list)

        # Find the target icd code from the icd code dataframe
        target_code = loc(pd.concat((icd_9, icd_10)), "TITLE", "in", list(chapter_titles.values()))

        # Merge the icd code dataframe with the diagnosis dataframe
        diagnoses = pd.merge(diagnoses, target_code.rename(columns={"DIAGNOSIS_CODE": "icd_code"}))

        return diagnoses

    def read_patients(self):
        """
        Read patients.csv.gz and return target subject_id dataframe

        :return patients dataframe
        """
        print("Reading patients.csv.gz...")

        patients = pd.read_csv(self.hosp_path + 'patients.csv.gz', compression='gzip', header=0, sep=',',
                               low_memory=False)
        patients = loc(patients, 'subject_id', 'in', self.subject_list)
        return patients
