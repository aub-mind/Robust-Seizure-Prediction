## Augmenting DL with Adversarial Training for Robust Prediction of Epilepsy Seizure

This repository contains the code used for the journal paper titled "Augmenting DL with Adversarial Training for Robust Prediction of Epilepsy Seizures" by Hussein A., Djandji M. et al which was published at ACM Transactions on Computing for Healthcare. The paper can be found here: https://dl.acm.org/doi/abs/10.1145/3386580.

## Requirements

* h5py (2.9.0)
* hickle (3.4.5)
* matplotlib (3.1.1)
* mne (0.11.0)
* pandas (0.25.1)
* scikit-learn (0.21.3)
* scipy (1.1.0)
* tensorflow-gpu (1.14.0)

## Main Folders Description

- CHBMIT and FB: Raw dataset folders. 
- CHBMIT_cache and FB_cache: Prepared data folders.
- models/: Model source code.
- utils/: Helping modules to load and prepare the data.


## Quick start

1. Download the two datasets (CHBMIT and FB) and move them into their folders.
    - CHBMIT: http://physionet.org/physiobank/database/chbmit/
    - FB: http://epilepsy.uni-freiburg.de.
2. Run ```main.py```
```
python main.py --mode without_AE --dataset CHBMIT
```

## Model

![Alt text](images/proposed_approach.PNG?raw=true "proposed_approach")


## Generate a Sample of Adversarial Examples

- Run ```inspect_AE.py``` to generate sample figures of advesarial examples 
![Alt text](images/AE_noise.png?raw=true "AE")


## Results

![Alt text](images/tsne.PNG?raw=true "tsne")

## Contacts

- [Amir Hussein](https://github.com/AmirHussein96) anh21@mail.aub.edu 
- [Marc Djandji] mgd10@mail.aub.edu

## Paper:

Cite our paper as:
```
@article{10.1145/3386580,
author = {Hussein, Amir and Djandji, Marc and Mahmoud, Reem A. and Dhaybi, Mohamad and Hajj, Hazem},
title = {Augmenting DL with Adversarial Training for Robust Prediction of Epilepsy Seizures},
year = {2020},
issue_date = {June 2020},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {1},
number = {3},
issn = {2691-1957},
url = {https://doi.org/10.1145/3386580},
doi = {10.1145/3386580},
journal = {ACM Trans. Comput. Healthcare},
month = jun,
articleno = {18},
numpages = {18},
keywords = {epileptic seizure prediction, multitask learning, Deep learning, adversarial examples, electroencephalogram}
}

```

