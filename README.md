Speech Emotion Recognition using CNN

Table of Contents

	1.	Overview
	2.	Features
	3.	Dataset
	4.	Requirements
	5.	Usage
	6.	Model Architecture
	7.	Results

1. Overview

This project implements Speech Emotion Recognition (SER) using a Convolutional Neural Network (CNN). The goal is to classify emotions from audio recordings based on their features. The model is trained to detect emotions such as happiness, sadness, anger, and neutrality from audio signals. The core steps involve:
	•	Extracting relevant features from speech data (e.g., MFCCs)
	•	Training a CNN model to recognize patterns in the extracted features
	•	Evaluating the model’s performance on a test dataset

2. Features

	•	Audio Feature Extraction: Uses Librosa for extracting Mel-Frequency Cepstral Coefficients (MFCCs).
	•	Visualization: Displays waveforms and spectrograms of audio samples for analysis.
	•	Model Training: A CNN architecture is used for training and classification.
	•	Performance Evaluation: Includes accuracy metrics and visualizations of training history.

3. Dataset

The script expects a dataset of labeled audio files representing various emotions. You may use publicly available datasets such as:
	•	RAVDESS: Ryerson Audio-Visual Database of Emotional Speech and Song
	•	TESS: Toronto Emotional Speech Set
	•	CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset
  •	SAVEE: Surrey Audio-Visual Expressed Emotion

4. Requirements

To run this project, you need the following Python packages:
	•	Python 3.8+
	•	NumPy
	•	Pandas
	•	Librosa
	•	Matplotlib
	•	Plotly
	•	Scikit-learn
	•	TensorFlow / Keras

5. Usage

	1.	Mount Google Drive (if using Google Colab):
The script starts by mounting Google Drive to access the dataset. 

	2.	Run the Script:
Make sure your dataset is ready and the environment is set up. 

	3.	Training:
The script includes functions for data preprocessing, model training, and evaluation. Adjust hyperparameters as needed within the script.

	5.	Testing:
The trained model can be tested on a separate set of audio files to evaluate its performance.

6. Model Architecture

The CNN model is structured as follows:
	•	Input Layer: Takes MFCC features as input
	•	Convolutional Layers: Feature extraction from input data
	•	Pooling Layers: Dimensionality reduction
	•	Flatten Layer: Flattening the 2D matrix to a vector
	•	Dense Layers: Fully connected layers for classification
	•	Output Layer: Softmax activation to predict emotion categories

7. Results

The script includes code for visualizing training accuracy and loss over epochs. Example metrics:
	•	Training Accuracy: Typically ranges between 70-85% depending on the dataset and model configuration
	•	Validation Accuracy: Similar range, indicating the model’s generalization capability
