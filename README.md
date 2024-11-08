{\rtf1\ansi\ansicpg1252\cocoartf2818
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 .SFNS-Bold;\f1\fnil\fcharset0 .SFNS-Regular;\f2\froman\fcharset0 TimesNewRomanPSMT;
\f3\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;\red14\green14\blue14;}
{\*\expandedcolortbl;;\cssrgb\c6700\c6700\c6700;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\sl324\slmult1\pardirnatural\partightenfactor0

\f0\b\fs54 \cf2 Speech Emotion Recognition using CNN
\f1\b0\fs38 \
\
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\sl324\slmult1\pardirnatural\partightenfactor0

\f0\b\fs44 \cf2 Table of Contents
\f1\b0\fs38 \
\
\pard\tqr\tx260\tx420\li420\fi-420\sl324\slmult1\sb240\partightenfactor0

\f2\fs28 \cf2 	1.	
\f1\fs38 Overview\

\f2\fs28 	2.	
\f1\fs38 Features\

\f2\fs28 	3.	
\f1\fs38 Dataset\

\f2\fs28 	4.	
\f1\fs38 Requirements\

\f2\fs28 	5.	
\f1\fs38 Usage\

\f2\fs28 	6.	
\f1\fs38 Model Architecture\

\f2\fs28 	7.	
\f1\fs38 Results\

\f2\fs28 	8.	
\f1\fs38 Contributing\

\f2\fs28 	9.	
\f1\fs38 License
\f3\fs24 \cf0 \
\
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\sl324\slmult1\pardirnatural\partightenfactor0

\f0\b\fs44 \cf2 1. Overview
\f1\b0\fs38 \
\
This project implements Speech Emotion Recognition (SER) using a Convolutional Neural Network (CNN). The goal is to classify emotions from audio recordings based on their features. The model is trained to detect emotions such as happiness, sadness, anger, and neutrality from audio signals. The core steps involve:\
\pard\tqr\tx100\tx260\li260\fi-260\sl324\slmult1\sb240\partightenfactor0
\cf2 	\'95	Extracting relevant features from speech data (e.g., MFCCs)\
	\'95	Training a CNN model to recognize patterns in the extracted features\
	\'95	Evaluating the model\'92s performance on a test dataset\
\
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\sl324\slmult1\pardirnatural\partightenfactor0

\f0\b\fs44 \cf2 2. Features
\f1\b0\fs38 \
\
\pard\tqr\tx100\tx260\li260\fi-260\sl324\slmult1\sb240\partightenfactor0
\cf2 	\'95	Audio Feature Extraction: Uses Librosa for extracting Mel-Frequency Cepstral Coefficients (MFCCs).\
	\'95	Visualization: Displays waveforms and spectrograms of audio samples for analysis.\
	\'95	Model Training: A CNN architecture is used for training and classification.\
	\'95	Performance Evaluation: Includes accuracy metrics and visualizations of training history.\
\
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\sl324\slmult1\pardirnatural\partightenfactor0

\f0\b\fs44 \cf2 3. Dataset
\f1\b0\fs38 \
\
The script expects a dataset of labeled audio files representing various emotions. You may use publicly available datasets such as:\
\pard\tqr\tx100\tx260\li260\fi-260\sl324\slmult1\sb240\partightenfactor0
\cf2 	\'95	RAVDESS: Ryerson Audio-Visual Database of Emotional Speech and Song\
	\'95	TESS: Toronto Emotional Speech Set\
	\'95	CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset\
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\pardirnatural\partightenfactor0

\f3\fs24 \cf0 \
\
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\sl324\slmult1\pardirnatural\partightenfactor0

\f0\b\fs44 \cf2 4. Requirements
\f1\b0\fs38 \
\
To run this project, you need the following Python packages:\
\pard\tqr\tx100\tx260\li260\fi-260\sl324\slmult1\sb240\partightenfactor0
\cf2 	\'95	Python 3.8+\
	\'95	NumPy\
	\'95	Pandas\
	\'95	Librosa\
	\'95	Matplotlib\
	\'95	Plotly\
	\'95	Scikit-learn\
	\'95	TensorFlow / Keras
\f3\fs24 \cf0 \
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\pardirnatural\partightenfactor0
\cf0 \
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\sl324\slmult1\pardirnatural\partightenfactor0

\f0\b\fs44 \cf2 5. Usage
\f1\b0\fs38 \
\
\pard\tqr\tx260\tx420\li420\fi-420\sl324\slmult1\sb240\partightenfactor0

\f2\fs28 \cf2 	1.	
\f0\b\fs38 Mount Google Drive
\f1\b0  (if using Google Colab):\
The script starts by mounting Google Drive to access the dataset. 
\f3\fs24 \cf0 \
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\pardirnatural\partightenfactor0
\cf0 \
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\sl324\slmult1\pardirnatural\partightenfactor0

\f1\fs38 \cf2 \
\pard\tqr\tx260\tx420\li420\fi-420\sl324\slmult1\sb240\partightenfactor0

\f2\fs28 \cf2 	2.	
\f0\b\fs38 Run the Script
\f1\b0 :\
Make sure your dataset is ready and the environment is set up. 
\f3\fs24 \cf0 \
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\pardirnatural\partightenfactor0
\cf0 \
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\sl324\slmult1\pardirnatural\partightenfactor0

\f1\fs38 \cf2 \
\pard\tqr\tx260\tx420\li420\fi-420\sl324\slmult1\sb240\partightenfactor0

\f2\fs28 \cf2 	3.	
\f0\b\fs38 Training
\f1\b0 :\
The script includes functions for data preprocessing, model training, and evaluation. Adjust hyperparameters as needed within the script.\

\f2\fs28 	4.	
\f0\b\fs38 Testing
\f1\b0 :\
The trained model can be tested on a separate set of audio files to evaluate its performance.\
\
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\sl324\slmult1\pardirnatural\partightenfactor0

\f0\b\fs44 \cf2 6. Model Architecture
\f1\b0\fs38 \
\
The CNN model is structured as follows:\
\pard\tqr\tx100\tx260\li260\fi-260\sl324\slmult1\sb240\partightenfactor0
\cf2 	\'95	Input Layer: Takes MFCC features as input\
	\'95	Convolutional Layers: Feature extraction from input data\
	\'95	Pooling Layers: Dimensionality reduction\
	\'95	Flatten Layer: Flattening the 2D matrix to a vector\
	\'95	Dense Layers: Fully connected layers for classification\
	\'95	Output Layer: Softmax activation to predict emotion categories\
\
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\sl324\slmult1\pardirnatural\partightenfactor0

\f0\b\fs44 \cf2 7. Results
\f1\b0\fs38 \
\
The script includes code for visualizing training accuracy and loss over epochs. Example metrics:\
\pard\tqr\tx100\tx260\li260\fi-260\sl324\slmult1\sb240\partightenfactor0
\cf2 	\'95	Training Accuracy: Typically ranges between 70-85% depending on the dataset and model configuration\
	\'95	Validation Accuracy: Similar range, indicating the model\'92s generalization capability\
	\'95	Ensure you have adequate computational resources (e.g., GPU) for training the CNN model, as it can be computationally intensive.}