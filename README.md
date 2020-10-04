# speech-emotion-recognition

A simple MLP classifier trained on the [RAVDESS dataset](https://smartlaboratory.org/ravdess/).

### [See notebook for a narrated walk-through of the code](https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/mlp_speech_classification.ipynb)

# Table of Contents
  - [Intro: The MLP classifier and RAVDESS dataset](https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/mlp_speech_classification.ipynb#Intro:-The-MLP-classifier-and-RAVDESS-dataset)
    - [At a Glance](https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/mlp_speech_classification.ipynb#At-a-Glance)
  - [Feature Engineering](https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/mlp_speech_classification.ipynb#Feature-Engineering)
    - [Short-Time Fourier Transform (STFT)](https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/mlp_speech_classification.ipynb#Short-Time-Fourier-Transform-(STFT))
    - [MFCC](https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/mlp_speech_classification.ipynb#MFCC)
    - [Mel Spectrograms and Mel-Frequency Cepstrums](https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/mlp_speech_classification.ipynb#Mel-Spectrograms-and-Mel-Frequency-Cepstrums)
    - [The Chromagram](https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/mlp_speech_classification.ipynb#The-Chromagram)
  - [Feature Extraction](https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/mlp_speech_classification.ipynb#Feature-Extraction)
    - [Load the Dataset and Compute its Features](https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/mlp_speech_classification.ipynb#Load-the-Dataset-and-Compute-its-Features)
  - [Training an MLP Classifier on a Train/Test Split](https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/mlp_speech_classification.ipynb#Training-an-MLP-Classifier-on-a-Train/Test-Split)
    - [K-Fold Cross-Validation](https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/mlp_speech_classification.ipynb#K-Fold-Cross-Validation)
    - [The Validation Curve: Further Tuning of Hyperparameters](https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/mlp_speech_classification.ipynb#The-Validation-Curve:-Further-Tuning-of-Hyperparameters)
    - [The Learning Curve: Determining Optimal Training Set Size](https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/mlp_speech_classification.ipynb#The-Learning-Curve:-Determining-Optimal-Training-Set-Size)
  - [Alternative Models](https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/mlp_speech_classification.ipynb#Alternative-Models)
      - [Linear model: Support Vector Classifier](https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/mlp_speech_classification.ipynb#Linear-model:-Support-Vector-Classifier)
      - [Non-Linear Decision Tree Ensemble Model: Random Forest](https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/mlp_speech_classification.ipynb#Non-Linear-Decision-Tree-Ensemble-Model:-Random-Forest)
  - [Conclusion](https://nbviewer.jupyter.org/github/IliaZenkov/speech-emotion-recognition/blob/master/mlp_speech_classification.ipynb#Conclusion)
