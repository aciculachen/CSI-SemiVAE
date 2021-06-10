# Semi-Supervised Learning with VAEs for Device-Free Fingerprinting Indoor Localization
######  Last update: 6/10/2021
## Introduction:
Implementation of semi-supervised variational auto-encoder (VAE) for Device Free Wi-Fi Fingerprinting Indoor Localization. 

The code is inherited and modified from [here](https://github.com/bjlkeng/sandbox/tree/master/notebooks/vae-semi_supervised_learning).
## Concept:
<img src="https://github.com/aciculachen/CSI-SemiVAE/blob/master/sVAE.png" width="600">

## Features:

- main.py: train the VAE model under the pre-defined indoor localization scenarios.
- generate_CSI.py: Generate CSI samples with pretrained GAN model.
- plot_CSI.py: code for plotting CSI samples
- models.py: definde semisupervised VAE 
- dataset: pre-collected CSI samples save as pickle in the form of (X_train, y_train, X_tst, y_tst)
## Dependencies:
- tensorflow 1.13
- python 3.6
