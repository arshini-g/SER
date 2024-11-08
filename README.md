# Speech Emotion Recognition using CNN

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Requirements](#requirements)
5. [Model Architecture](#model-architecture)
6. [Results](#results)

## Overview
This project implements Speech Emotion Recognition (SER) using a Convolutional Neural Network (CNN). The goal is to classify emotions from audio recordings based on their features. The model is trained to detect emotions such as happiness, sadness, anger, and neutrality from audio signals. The core steps involve:
- Extracting relevant features from speech data (e.g., MFCCs)
- Training a CNN model to recognize patterns in the extracted features
- Evaluating the model’s performance on a test dataset

## Features
- **Audio Feature Extraction**: Used Librosa for extracting Mel-Frequency Cepstral Coefficients (MFCCs).
- **Visualization**: Displays waveforms and spectrograms of audio samples for analysis.
- **Model Training**: A CNN architecture is used for training and classification.
- **Performance Evaluation**: Includes accuracy metrics and visualizations of training history.

## Dataset
The script uses various datasets of labeled audio files representing various emotions such as:
- **RAVDESS**: Ryerson Audio-Visual Database of Emotional Speech and Song
- **TESS**: Toronto Emotional Speech Set
- **CREMA-D**: Crowd-sourced Emotional Multimodal Actors Dataset
- **SAVEE**: Surrey Audio-Visual Expressed Emotion

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Librosa
- Matplotlib
- Plotly
- Scikit-learn
- TensorFlow / Keras

## Model Architecture
The CNN model is structured as follows:
- **Input Layer**: Takes MFCC and various features as input
- **Convolutional Layers**: Feature extraction from input data
- **Pooling Layers**: Dimensionality reduction
- **Flatten Layer**: Flattening the 2D matrix to a vector
- **Dense Layers**: Fully connected layers for classification
- **Output Layer**: Softmax activation to predict emotion categories

## Results
The script includes code for visualizing training accuracy and loss over epochs. Example metrics:
- **Training Accuracy**: Typically ranges between 92-95%
- **Validation Accuracy**: Similar range, indicating the model’s generalization capability
