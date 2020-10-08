# speech-emotion-recognition

A simple MLP classifier trained on the [RAVDESS dataset](https://smartlaboratory.org/ravdess/).

### [See notebook for a narrated walk-through of the code](https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/audio_classification.ipynb)

<!--TABLE OF CONTENTS-->
# Table of Contents
  - [Intro: Speech Emotion Recognition on the RAVDESS dataset](#https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/audio_classification.ipynb#Intro:-Speech-Emotion-Recognition-on-the-RAVDESS-dataset)
  - [Machine Learning Process Overview](#https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/audio_classification.ipynb#Machine-Learning-Process-Overview)
  - [Feature Engineering](#https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/audio_classification.ipynb#Feature-Engineering)
    - [Short-Time Fourier Transform](#https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/audio_classification.ipynb#Short-Time-Fourier-Transform)
    - [Mel-Frequency Cepstral Coefficients](#https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/audio_classification.ipynb#Mel-Frequency-Cepstral-Coefficients)
    - [Mel Spectrograms and Mel-Frequency Cepstrums](https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/audio_classification.ipynb#Mel-Spectrograms-and-Mel-Frequency-Cepstrums)
    - [The Chromagram](#https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/audio_classification.ipynb#The-Chromagram)
  - [Feature Extraction](#https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/audio_classification.ipynb#Feature-Extraction)
    - [Load the Dataset and Compute Features](#https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/audio_classification.ipynb#Load-the-Dataset-and-Compute-Features)
    - [Feature Scaling](#https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/audio_classification.ipynb#Feature-Scaling)
  - [Classical Machine Learning Models](#https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/audio_classification.ipynb#Classical-Machine-Learning-Models)
    - [Training: The 80/20 Split and Validation](#https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/audio_classification.ipynb#Training:-The-80/20-Split-and-Validation)
    - [Comparing Models](#https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/audio_classification.ipynb#Comparing-Models)
    - [The Support Vector Machine Classifier](#https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/audio_classification.ipynb#The-Support-Vector-Machine-Classifier)
    - [k Nearest Neighbours](#https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/audio_classification.ipynb#k-Nearest-Neighbours)
    - [Random Forests](#https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/audio_classification.ipynb#Random-Forests)
        - [OOB Score](#https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/audio_classification.ipynb#OOB-Score)
    - [Next Steps](#Next-Steps)
  - [The MLP Model for Classification](#https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/audio_classification.ipynb#The-MLP-Model-for-Classification)
    - [Choice of Hyperparameters](#https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/audio_classification.ipynb#Choice-of-Hyperparameters)
    - [Network Architecture](#https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/audio_classification.ipynb#Network-Architecture)
    - [Hyperparameter Optimization and Grid Search](#https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/audio_classification.ipynb#Hyperparameter-Optimization-and-Grid-Search)
  - [Training and Evaluating the MLP Model](#https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/audio_classification.ipynb#Training-and-Evaluating-the-MLP-Model)
    - [K-Fold Cross-Validation](#https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/audio_classification.ipynb#K-Fold-Cross-Validation)
    - [The Validation Curve: Further Tuning of Hyperparameters](#https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/audio_classification.ipynb#The-Validation-Curve:-Further-Tuning-of-Hyperparameters)
    - [The Learning Curve: Determining Optimal Training Set Size](#https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/audio_classification.ipynb#The-Learning-Curve:-Determining-Optimal-Training-Set-Size)
  - [Conclusion](#https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/audio_classification.ipynb#Conclusion)
