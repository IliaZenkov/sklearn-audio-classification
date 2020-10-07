# speech-emotion-recognition

A simple MLP classifier trained on the [RAVDESS dataset](https://smartlaboratory.org/ravdess/).

### [See notebook for a narrated walk-through of the code](https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/audio_classification.ipynb)

<!--TABLE OF CONTENTS-->
# Table of Contents
  - [Intro - Speech Emotion Recognition on the RAVDESS dataset](#Intro:-Speech-Emotion-Recognition-on-the-RAVDESS-dataset)
  - [Machine Learning Process Overview](#Machine-Learning-Process-Overview)
  - [Feature Engineering](#Feature-Engineering)
    - [Short-Time Fourier Transform](#Short-Time-Fourier-Transform)
    - [Mel-Frequency Cepstral Coefficients](#Mel-Frequency-Cepstral-Coefficients)
    - [Mel Spectrograms and Mel-Frequency Cepstrums](#Mel-Spectrograms-and-Mel-Frequency-Cepstrums)
    - [The Chromagram](#The-Chromagram)
  - [Feature Extraction](#Feature-Extraction)
    - [Load the Dataset and Compute Features](#Load-the-Dataset-and-Compute-Features)
    - [Feature Scaling](#Feature-Scaling)
  - [Classical Machine Learning Models](#Classical-Machine-Learning-Models)
    - [Training: The 80/20 Split and Validation](#Training:-The-80/20-Split-and-Validation)
    - [Comparing Models](#Comparing-Models)
    - [The Support Vector Machine Classifier](#The-Support-Vector-Machine-Classifier)
    - [k Nearest Neighbours](#k-Nearest-Neighbours)
    - [Random Forests](#Random-Forests)
        - [OOB Score](#OOB-Score)
    - [Next Steps](#Next-Steps)
  - [The MLP Model for Classification](#The-MLP-Model-for-Classification)
    - [Choice of Hyperparameters](#Choice-of-Hyperparameters)
    - [Network Architecture](#Network-Architecture)
    - [Hyperparameter Optimization: Grid Search](#Hyperparameter-Optimization:-Grid-Search)
  - [Training and Evaluating the MLP Model](#Training-and-Evaluating-the-MLP-Model)
    - [K-Fold Cross-Validation](#K-Fold-Cross-Validation)
    - [The Validation Curve: Further Tuning of Hyperparameters](#The-Validation-Curve:-Further-Tuning-of-Hyperparameters)
    - [The Learning Curve: Determining Optimal Training Set Size](#The-Learning-Curve:-Determining-Optimal-Training-Set-Size)
  - [Conclusion](#Conclusion)
