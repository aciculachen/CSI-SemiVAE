# Semi-Supervised Learning with VAEs for Device-Free Fingerprinting Indoor Localization
######  Last update: 8/25/2021
## Introduction:
Implementation of semi-supervised variational auto-encoder (VAE) for Device Free Wi-Fi Fingerprinting Indoor Localization. 

For more details and evaluation results, please check out our original [paper](https://www.citi.sinica.edu.tw/papers/rchang/8092-F.pdf), which will be published in IEEE GLOBECOM 2021.

The code is inherited and modified from [here](https://github.com/bjlkeng/sandbox/tree/master/notebooks/vae-semi_supervised_learning).
## Concept:
<img src="https://github.com/aciculachen/CSI-SemiVAE/blob/master/sVAE.png" width="600">

## Features:

- **main.py**: train the VAE model under the pre-defined indoor localization scenarios.
- **plot_CSI.py**: code for plotting CSI samples
- **models.py**: definde semisupervised VAE 
- **dataset**: pre-collected CSI samples save as pickle in the form of (X_train, y_train, X_tst, y_tst)
## Dependencies:
- tensorflow 1.13
- python 3.6.4
