from icd9cms.icd9 import search
import numpy as np

icd_chapter = {
    "infectious_and_parasitic_disease": np.arange(1, 140, 1),
    "neoplasms": np.arange(140, 240, 1),
    "endocrine_nutritional_and_metabolic_disease_and_immunity_disorders": np.arange(240, 280, 1),
    "diseases_of_the_blood_and_blood_forming_organs": np.arange(280, 290, 1),
    "mental_disorders": np.arange(290, 320, 1),
    "diseases_of_the_nervous_system_and_sense_organs": np.arange(320, 390, 1),
    "diseases_of_the_circulatory_system": np.arange(390, 460, 1),
    "diseases_of_the_respiratory_system": np.arange(460, 520, 1),
    "diseases_of_the_digestive_system": np.arange(520, 580, 1),
    "diseases_of_the_genitourinary_system": np.arange(580, 630, 1),
    "complications_of_pregnancy_childbirth_and_the_puerperium": np.arange(630, 680, 1),
    "diseases_of_the_skin_and_subcutaneous_tissue": np.arange(680, 710, 1),
    "diseases_of_the_musculoskeletal_system_and_connective_tissue": np.arange(710, 740, 1),
    "congenital_anomalies": np.arange(740, 760, 1),
    "certain_conditions_originating_in_the_perinatal_period": np.arange(760, 780, 1),
    "symptoms_signs_and_ill_defined_conditions": np.arange(780, 800, 1),
    "injury_and_poisoning": np.arange(800, 1000, 1)
}

circulatory_system_disease = {
    "acute_rheumatic_heart_disease": np.arange(390, 393, 1),
    "chronic_rheumatic_heart_disease": np.arange(393, 398, 1),
    "hypertensive_disease": np.arange(401, 406, 1),
    "ischemic_heart_disease": np.arange(410, 415, 1),
    "pulmonary_circulation_disease": np.arange(415, 418, 1),
    "other_forms_of_heart_disease": np.arange(420, 430, 1),
    "cerebrovascular_disease": np.arange(430, 439, 1),
    "disease_of_arteries_arterioles_capillaries": np.arange(440, 450, 1),
    "disease_of_veins_lymphatics_and_other": np.arange(451, 460, 1)
}

icd = {
    "circulatory": circulatory_system_disease,
}

def get_parent_icd_code(parent_code_dict):
    p_icd = []
    for key, value in parent_code_dict.items():
        p_icd.extend(value)
    return p_icd


def get_children_icd_code(list_of_parent_code):
    c_icd = []
    non_leaf = []
    for code in list_of_parent_code:
        c_icd.append(str(code))
        children = search(str(code)).children
        if children is not None:
            for c in children:
                c_icd.append(c.code)
                if c.children is not None:
                    non_leaf.append(c)
                    c_icd.extend([x.code for x in c.children])
                # else:
                #     c_icd.append(c.code)
            # c_icd.extend([x.code for x in children])
    # for n in non_leaf:
    #     c_icd.append(n.code)
    return c_icd

def get_code_from_icd_dict(class_name):
    parent_code = get_parent_icd_code(icd[class_name])
    return get_children_icd_code(parent_code)


if __name__ == "__main__":
    parent_code = get_parent_icd_code(circulatory_system_disease)
    test = get_children_icd_code(parent_code)
    print(len(test))
