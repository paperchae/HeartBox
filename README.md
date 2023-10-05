# HeartBox

## Description

- This project is to detect heart disease using biomedical signals such as **ECG**, **PPG** and **ABP** through [Pytorch](https://pytorch.org/).
- For Data,
  - _Data Inspection_
  - _Preprocessing_
  - _Feature Extraction_
  - _Augmentation_ is provided
- Training and Evaluation is provided
- This project will tie together all the functionality in [MIMIC](https://github.com/paperchae/MIMIC)
  and [CNIBP](https://github.com/remotebiosensing/rppg/tree/main/cnibp), so that it can be used as a single package for
  heart disease detection.

***

## Preprocessing

[preprocessing](/preprocessing)
> - Baseline Correction
>  - Removes baseline wandering created by respiration, body movement, etc.
> - Denoising
>  - Removes noise created by power line, muscle contraction, etc.

***

## Data Augmentation

[Augmentation](preprocessing/signal_augmentation.py)

***

## Datasets

### Vital Signal Datasets

> [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/)
>- 12 leads ECG, 10 seconds length
>- Patients: approximately *160,000*
>- Records: *800,000*
>
> [MIMIC-IV Waveform Database](https://physionet.org/content/mimic4wdb/0.1.0/)
>- *12 leads ECG*, 10 seconds length
>- Patients: *198*
>- Records: *200 (10,000 records upcomming)*
>
> [MIMIC-III Waveform Database](https://physionet.org/content/mimic3wdb/1.0/)
>- Contains *ECG, ABP, PPG, Resp, etc*.
>- Patients: approximately *30,000*
>- Records: *67,830*
>
>[PTB Diagnostic ECG Database](https://physionet.org/content/ptbdb/1.0.0/)
>- TBD

### Clinical Database

> [MIMIC-IV](https://physionet.org/content/mimiciv/2.2/)
> - Patients
***

## Results

[Results](/results)

[mindmeister](https://mm.tt/app/map/2924715348?t=LSpBskuWaX)

## Contact

JONGEUI CHAE : forownsake@gmail.com

Please feel free to contact, open for any collaboration.
