import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from preprocess.icd.icd_chapter import icd_chapters


class ICD:
    """
    ICD code retrieving class

    Support version 9 and 10
    Used for filtering patients with ICD Chapter titles (e.g. circulatory, respiratory, etc.)

    """
    def __init__(self, version: int=9):
        self.version = version
        self.data_path = os.path.join(os.path.dirname(__file__), "result/icd_{}.csv".format(self.version))
        self.flag = self.check_existence()
        self.icd_dict = icd_chapters["{}".format(self.version)]
        self.path = self.icd_dict["path"]
        self.target_dict = self.icd_dict["target_dict"]

    def check_existence(self):
        """
        Check preprocessed ICD csv existence
        """

        return os.path.isfile(self.data_path)

    def get_dataframe(self):
        """
        If ICD csv exist read csv file.
        Otherwise, create csv files with generate_icd_df() and return icd dataframe
        """
        if self.flag:
            df = pd.read_csv(self.data_path)
        else:
            print("Generating icd_{}.csv...".format(self.version))
            df = self.generate_icd_df()

        return df

    def _get_icd_block(self, icd_code):
        """
        According to version of ICD code, extract chapter information from icd_code.

        * need to work on blocks with E, V in icd 9
        """
        if self.version == 9:
            block = icd_code[:3]
            # if block[0] in ['E', 'V']:
            #     block = icd_code[:4]
        elif self.version == 10:
            block = icd_code[:1]
        else:
            raise ValueError("Invalid version")
        return block

    def _check_chapter(self, icd_code):
        """
        With block information from icd_code, retrieve chapter information

        * Used for generating csv files in exact format
        """
        icd_block = self._get_icd_block(icd_code)
        exist_flag = False

        for chapter, value in self.target_dict.items():
            if self.version == 9:
                blocks = [c[:3] for c in value["block"].split('-')]
                blocks = np.arange(int(blocks[0]), int(blocks[1]) + 1)
                blocks = ['{0:03d}'.format(b) for b in blocks]
            elif self.version == 10:
                blocks = [c[:1] for c in value["block"].split('-')]
            else:
                raise ValueError("Invalid version")

            if icd_block in blocks:
                return chapter, value["title"]
            else:
                continue
        if not exist_flag:
            return "None", "None"

    def _generate_row(self, data):
        """
        Generate rows for dataframe

        * Used for fast execution making dataframe.
        """
        if self.version == 9:
            df = pd.DataFrame(data=[data],
                              columns=["CHAPTER", "DIAGNOSIS_CODE", "TITLE", "SHORT_DESC", "LONG_DESC"])
        elif self.version == 10:
            df = pd.DataFrame(data=[data],
                              columns=["CHAPTER", "DIAGNOSIS_CODE", "TITLE", "IS_HEADER", "SHORT_DESC", "LONG_DESC"])
        else:
            raise ValueError("Invalid version")

        return df

    def generate_icd_df(self):
        """
        Generates ICD Code csv files from text & xlsx files

        Including:
            CHAPTER : Roman order for chapter
            DIAGNOSIS_CODE : icd code
            TITLE : Text of disease categorie
            IS_HEADER : Header flag for disease, available only for version 10
            SHORT_DESC : Short text description of icd code with acronyms
            LONG_DESC : Long text description of icd code
        """
        df_list = []

        if self.version == 9:
            df = pd.read_excel(self.path)

            for idx, row in tqdm(df.iterrows(), total=len(df), desc="ICD 9 Data..."):
                icd_code = row["DIAGNOSIS CODE"]
                try:
                    chapter, title = self._check_chapter(icd_code)
                except:
                    chapter, title = "None", "None"
                short_desc = row["SHORT DESCRIPTION"]
                long_desc = row["LONG DESCRIPTION"]
                df_list.append(self._generate_row([chapter, icd_code, title, short_desc, long_desc]))

        elif self.version == 10:
            f = open(self.path, "r")
            lines = f.readlines()

            for l in tqdm(lines, total=len(lines), desc="ICD 10 Data..."):
                icd_code = l[6:14].strip()
                chapter, title = self._check_chapter(icd_code)
                is_header = True if l[14] == '0' else False
                short_desc = l[16:77].strip()
                long_desc = l[77:-1]
                df_list.append(self._generate_row([chapter, icd_code, title, is_header, short_desc, long_desc]))

        else:
            raise ValueError("Invalid version")

        icd_df = pd.concat(df_list, ignore_index=True)

        if not self.flag:
            icd_df.to_csv("result/icd_{}.csv".format(self.version), index=False)

        return icd_df


def retrieve_chapter_title(target_title):
    """
    Find the chapter title from the target diagnosis per icd version
    :param target_title: target diagnosis

    :return: chapter title related to the target diagnosis title
    """
    version = [x["target_dict"] for x in icd_chapters.values()]
    title = {"9": 0, "10": 0}
    version_cnt = 9

    # Search related chapter title from icd chapters
    for v in version:
        for chap, value in v.items():
            if target_title.lower() in value["title"].lower():
                title[str(version_cnt)] = value["title"]
                break
            else:
                continue
        version_cnt += 1

    # If there is no matched chapter title with target title, raise ValueError
    if all(list(title.values())):
        return title
    else:
        raise ValueError("Invalid chapter, check icd_chapter.py")


if __name__ == "__main__":
    icd_9_df = ICD(version=9).get_dataframe()
    icd_10_df = ICD(version=10).get_dataframe()

    print("ICD 9")
    print(icd_9_df.head())
    print("ICD 10")
    print(icd_10_df.head())
