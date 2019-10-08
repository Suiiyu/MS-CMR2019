# MS-CMR2019
Code for the [2019 Multi-sequence Cardiac MR Segmentation Challenge](https://zmiclab.github.io/mscmrseg19/). The main purpose of this challenge is segmenting left ventricle (LV), right ventricle (RV) and left ventricle myocardium (LVM) from LGE CMR data. It published 45 patients CMR data with three modalities, T2, b-SSFP, and LGE. There are 35 labeled T2 CMR data with about 3 slices of each patient, and 35 labeled b-SSFP CMR data with about 11 slices of each patient, and just 5 labeled LGE CMR data with about 15 slices of each patient. The rest volumes are unlabeled data. The rarely labeled target data increases the challenge sharply. For this challenge, we proposed an automatic segmentation framework. The paper is available at [arXiv](https://arxiv.org/abs/1909.05488).

The pipeline of our method is show below:

<p align="center">
    <img src="images/Fig1.Framework.png" width="1000">
</p>

## Requirements

Python 3.5

Keras based on Tensorflow

