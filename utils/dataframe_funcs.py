import pandas as pd
import numpy as np

# import pandas as pd
from functools import reduce

# from Analytics.Statistics import DescriptiveStatistics as ds
import matplotlib.pyplot as plt

# import wfdb
import gzip
import shutil
import os

raw_data_path = "/hdd/hdd1/dataset/mimic-iii-clinical-database-1.4/"  # ubuntu


def uom_converter(inputs, measure_type):
    """
    Convert unit of measure
    :param inputs:
        measurement values in inch, lb, fahrenheit...
    :param measure_type:
        Height, Weight, Temperature
    :return:
        converted values in cm, kg, celsius...
    """
    if measure_type == "Height":
        return np.array(inputs.astype("float")) * 2.54
    elif measure_type == "Weight":
        return np.array(inputs.astype("float")) * 0.453592
    elif measure_type == "Temperature":
        return (np.array(inputs.astype("float")) - 32) * 5 / 9


def get_item_id(label):
    """
    :param label:
        Patient's physical information
        e.g. Height, Weight, Temperature...
    :return items:
        Dataframe of ITEMID, LABEL, ABBREVIATION, DBSOURCE, LINKSTO, CATEGORY, UNITNAME, PARAM_TYPE, CONCEPTID
    """

    items = pd.read_csv(
        raw_data_path + "D_ITEMS.csv.gz",
        compression="gzip",
        header=0,
        sep=",",
        quotechar='"',
    )

    items = items[items["LABEL"].str.contains(label, na=False)]
    if label == "Height":
        items = items.loc[list(items["LABEL"] != "Height of Bed")]
    if label == "Weight":
        items = items.loc[list(items["LABEL"] != "Weight Change  (gms)")]

    return items


def get_physical_df(df, target):
    """

    Extract patient's physical information using ITEMID
    - Converts unit of measure
        -> Height: inch to cm
        -> Weight: lb to kg
        -> Temperature: fahrenheit to Celsius

    :param df:
        Chunk dataframe from CHARTEVENTS.csv.gz
    :param target:
        Patient's physical information
    :return:
        Dataframe of patient's physical information
    """
    item_df = get_item_id(target)
    height_list = list(item_df["ITEMID"])
    templist = []
    for h in height_list:
        test = df[df["ITEMID"] == h][
            ["SUBJECT_ID", "HADM_ID", "CHARTTIME", "ITEMID", "VALUE", "VALUEUOM"]
        ]
        if len(test) == 0:
            continue

        if target == "Height":
            if test["VALUEUOM"].tolist()[0] in ["In", "Inch", "inch"]:
                # test.loc[test['VALUEUOM'].str.contains('In|Inch|inch'), 'VALUE'] = test['VALUE'].astype('float') * 2.54
                test.loc[
                    test["VALUEUOM"].str.contains("In|Inch|inch"), "VALUE"
                ] = uom_converter(test["VALUE"], target)
                test.loc[
                    test["VALUEUOM"].str.contains("In|Inch|inch"), "VALUEUOM"
                ] = "cm"
        elif target == "Weight":
            if test["VALUEUOM"].tolist()[0] in ["lb", "pound", "pounds", "lbs"]:
                test.loc[
                    test["VALUEUOM"].str.contains("lb|pound|pounds|lbs"), "VALUE"
                ] = uom_converter(test["VALUE"], target)
                test.loc[
                    test["VALUEUOM"].str.contains("lb|pound|pounds|lbs"), "VALUEUOM"
                ] = "kg"
            else:
                if (
                    item_df[item_df["ITEMID"] == h]["LABEL"]
                    .str.contains("lbs")
                    .tolist()[0]
                ):
                    test["VALUE"] = uom_converter(test["VALUE"], target)
                    test["VALUEUOM"] = "kg"

        # else:
        #     raise NotImplementedError('Not implemented yet')
        test["VALUE"] = test["VALUE"].astype("float").__round__(2)
        templist.append(test)

    if len(templist) == 1:
        return templist[0].drop_duplicates(["SUBJECT_ID"], keep="last")
    elif len(templist) == 0:
        return pd.DataFrame(
            columns=[
                "SUBJECT_ID",
                "HADM_ID",
                "CHARTTIME",
                "ITEMID",
                "VALUE",
                "VALUEUOM",
            ]
        )
    else:
        return pd.concat(templist).drop_duplicates(["SUBJECT_ID"], keep="last")


def read_chartevent(subject_list, debug=False):
    """

    Read CHARTEVENTS.csv.gz and extract subject's chart information

    :param debug:
    :param subject_list:
    :return:
    """
    chunk_size = 10 ** 6
    cnt = 0
    height_list = []
    weight_list = []
    sbp_list = []
    dbp_list = []
    # aortic_sys_list = []
    # aortic_dia_list = []
    for chunk in pd.read_csv(
        raw_data_path + "CHARTEVENTS.csv.gz",
        chunksize=chunk_size,
        compression="gzip",
        header=0,
        sep=",",
        quotechar='"',
        low_memory=False,
    ):
        chunk = loc(chunk, "SUBJECT_ID", "in", subject_list)
        if len(chunk) == 0:
            print("no data")
            continue
        height_list.append(get_physical_df(chunk, "Height"))
        weight_list.append(get_physical_df(chunk, "Weight"))
        sbp_list.append(get_physical_df(chunk, "Systolic"))
        dbp_list.append(get_physical_df(chunk, "Diastolic"))
        # aortic_sys_list.append(get_physical_df(chunk, 'Aortic Pressure Signal - Systolic'))
        # aortic_dia_list.append(get_physical_df(chunk, 'Aortic Pressure Signal - Diastolic'))
        if debug:
            cnt += 1
            print(cnt)
            if cnt == 10:
                break
    total_list = [height_list, weight_list, sbp_list, dbp_list]
    total_list = [pd.concat(x)[["SUBJECT_ID", "VALUE"]] for x in total_list]
    total_df = reduce(
        lambda x, y: pd.merge(x, y, on="SUBJECT_ID", how="outer"), total_list
    )
    total_df.columns = ["SUBJECT_ID", "HEIGHT", "WEIGHT", "SBP", "DBP"]

    return total_df.drop_duplicates("SUBJECT_ID", keep="last")


def loc(df, column, condition, value):
    if condition == "eq":
        return df.loc[df[column] == value]
    elif condition == "in":
        return df.loc[df[column].isin(value)]
    elif condition == "not in":
        return df.loc[~df[column].isin(value)]


def read_patients(subject_list, debug=False):
    patient_df = pd.read_csv(
        raw_data_path + "PATIENTS.csv.gz", compression="gzip", header=0, sep=","
    )
    patient_df = loc(patient_df, "SUBJECT_ID", "in", subject_list)[
        ["SUBJECT_ID", "GENDER", "DOB", "EXPIRE_FLAG"]
    ]
    admission_df = pd.read_csv(
        raw_data_path + "ADMISSIONS.csv.gz", compression="gzip", header=0, sep=","
    )
    admission_df = loc(admission_df, "SUBJECT_ID", "in", subject_list)[
        [
            "SUBJECT_ID",
            "HADM_ID",
            "ADMITTIME",
            "ETHNICITY",
            "DIAGNOSIS",
            "HOSPITAL_EXPIRE_FLAG",
            "HAS_CHARTEVENTS_DATA",
        ]
    ]
    merged_info_df = pd.merge(
        patient_df, admission_df, how="outer", on="SUBJECT_ID"
    ).drop_duplicates(["SUBJECT_ID"], keep="last")
    dob_year = merged_info_df["DOB"].str.split("-", expand=True)[0].astype(int)
    admit_year = merged_info_df["ADMITTIME"].str.split("-", expand=True)[0].astype(int)
    merged_info_df["AGE"] = admit_year - dob_year
    health_df = read_chartevent(subject_list, debug)

    merged_df = pd.merge(merged_info_df, health_df, how="outer", on="SUBJECT_ID")[
        [
            "SUBJECT_ID",
            "GENDER",
            "AGE",
            "HEIGHT",
            "WEIGHT",
            "ETHNICITY",
            "SBP",
            "DBP",
            "DIAGNOSIS",
            "EXPIRE_FLAG",
            "HOSPITAL_EXPIRE_FLAG",
        ]
    ]
    if not debug:
        merged_df.to_csv("./result/patients.csv")
    return merged_df


if __name__ == "__main__":
    debug_mode = True
    pid = pd.read_csv("result/pid.csv", header=0, sep=",").to_numpy(dtype=int)[:, -1]
    patients_df = read_patients(pid, debug_mode)
