import pandas as pd
from tqdm import tqdm
from utils.pandas_utils import *
import numpy as np
from icd_chapter import icd_9_chapters, icd_10_chapters


class ICD:
    def __init__(self, version: int):
        self.version = version
        if self.version == 9:
            self.path = "ICD-9-CM-v32/CMS32_DESC_LONG_SHORT_DX.xlsx"
            self.target_dict = icd_9_chapters
            self.df = self.generate_icd_df_9()
        elif self.version == 10:
            self.path = "ICD-10-CM/icd10cm_order_2024.txt"
            self.target_dict = icd_10_chapters
            self.df = self.generate_icd_df_10()

    def _get_icd_block(self, icd_code):
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
                exist_flag = True
                return chapter, value["title"]
            else:
                continue
        if not exist_flag:
            return "None", "None"

    def _generate_row(self, data):
        if self.version == 9:
            df = pd.DataFrame(data=[data],
                              columns=["CHAPTER", "DIAGNOSIS_CODE", "TITLE", "SHORT_DESC", "LONG_DESC"])
        elif self.version == 10:
            df = pd.DataFrame(data=[data],
                              columns=["CHAPTER", "DIAGNOSIS_CODE", "TITLE", "IS_HEADER", "SHORT_DESC", "LONG_DESC"])
        else:
            raise ValueError("Invalid version")

        return df

    def generate_icd_df_10(self):
        f = open(self.path, "r")
        lines = f.readlines()

        df_list = []

        for l in tqdm(lines, total=len(lines)):
            icd_code = l[6:14].strip()
            chapter, title = self._check_chapter(icd_code)
            is_header = True if l[14] == '0' else False
            short_desc = l[16:77].strip()
            long_desc = l[77:-1]
            df_list.append(self._generate_row([chapter, icd_code, title, is_header, short_desc, long_desc]))

        icd_df = pd.concat(df_list, ignore_index=True)

        return icd_df

    def generate_icd_df_9(self):
        df = pd.read_excel(self.path)

        df_list = []

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            icd_code = row["DIAGNOSIS CODE"]
            try:
                chapter, title = self._check_chapter(icd_code)
            except:
                chapter, title = "None", "None"
            short_desc = row["SHORT DESCRIPTION"]
            long_desc = row["LONG DESCRIPTION"]
            df_list.append(self._generate_row([chapter, icd_code, title, short_desc, long_desc]))

        icd_df = pd.concat(df_list, ignore_index=True)
        return icd_df


if __name__ == "__main__":
    icd_9_df = ICD(version=9).df
    icd_9_df.to_csv("result/icd_9.csv", index=False)
    icd_10_df = ICD(version=10).df
    icd_10_df.to_csv("result/icd_10.csv", index=False)
