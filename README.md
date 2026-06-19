# CMCT: Cross-Modal Concept Transfer

This is an official implementation of our MICCAI 2026 paper.

Paper: [Cross-Modal Concept Transfer: From ECG Signals to Images for Explainable Disease Prediction](TBD)

## Environments
- Python:
- PyTorch:
- CUDA:
- cuDNN:
- GPU: NVIDIA RTX 3090, A100

### Installation
'''console
(base) user@server:~$ conda create -n cmct python=3.xx -y
(base) user@server:~$ conda activate cmct
(cmct) user@server:~$ git clone https://github.com/mldlcl2022/CMCT.git
(cmct) user@server:~$ cd CMCT
(cmct) user@server:~CMCT$ pip install -r requirements.txt
'''

### Requirements
- pytorch 2.xx.x
- 

## Datasets

### Data download
Datasets we used are as follows:
- **CPSC2018**: We used the [CPSC2018](http://2018.icbeb.org/Challenge.html) dataset. Since the dataset was not available for download from the official page at the time of our access, we downloaded the data from [PhysioNet](https://physionet.org/content/challenge-2020/1.0.2/training/#files-panel) and used it for our experiments.
- **PTB-XL**: We downloaded the [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) dataset from PhysioNet and used it in our experiments.

We used the predefined train, valid, and test split csv files released by the authors of "Zero-Shot ECG Classification with Multimodal Learning and Test-time Clinical Knowledge Enhancement." without modification. We did not create new data splits.

### ECG image preparation
To convert ECG signals into ECG images, we used [ECG-Image-Kit](https://github.com/alphanumericslab/ecg-image-kit), the official GitHub repository associated with the paper [“ECG-Image-Kit: a synthetic image generation toolbox to facilitate deep learning-based electrocardiogram digitization”](https://doi.org/10.1088/1361-6579/ad4954).

Using this toolkit, we generated ECG image datasets from the original ECG signal datasets for our experiments.

### ECG concept preparation
Run the following command to extract ECG concepts from the signal datasets. (This takes approximately 20 minutes in our environment.)

'''console
(cmct) user@server:~CMCT$ cd datasets
(cmct) user@server:~CMCT/datasets$ python ecg_concept_extractor.py --data_name <DATA_NAME> --root_dir <ROOT_DIR>
'''

### Expected directory structure
After downloading the datasets into the `signals` directory and completing ECG image and concept preparation, the expected directory structure is shown below using CPSC2018 as an example.

'''text
CMCT/
├── ...
└── datasets
    ├── cpsc2018
    |   ├── concepts
    |   |   ├── scaler.joblib
    |   |   ├── test.csv
    |   |   ├── train.csv
    |   |   └── valid.csv
    |   ├── images
    |   |   ├── g1
    |   |   |   ├── A0001.dat
    |   |   |   ├── A0001.hea
    |   |   |   ├── A0001-0.png
    |   |   |   └── ...
    |   |   ├── g2
    |   |   ├── ...
    |   |   └── g7
    |   └── signals
    |   |   ├── g1
    |   |   |   ├── A0001.hea
    |   |   |   ├── A0001.mat
    |   |   |   └── ...
    |   |   ├── g2
    |   |   ├── ...
    |   |   └── g7
    ├── ecg-image-kit
    ├── splits
    |   └── cpsc2018
    |   |   ├── test.csv
    |   |   ├── train.csv
    |   |   └── valid.csv
    ├── ecg_concept_extractor.py
    └── ecg_image_kit.py
'''