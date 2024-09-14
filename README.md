# Audio Classification Project Report

## 1. Introduction

This report outlines the development of an audio classification system designed to categorize audio samples into predefined classes. The project aims to create a robust classification model despite the constraints of a very small dataset, with only three samples per class.

## 2. Approach

Given the limited data, our approach focuses on maximizing the information extracted from each audio sample while using evaluation methods suitable for small datasets. The key steps in our approach are:

1. Feature extraction from audio files
2. Dataset creation
3. Model selection and training
4. Model evaluation
5. Final model selection and deployment

## 3. Feature Extraction

We selected a combination of time and frequency domain features to capture various aspects of the audio signals:

1. **Mel-frequency cepstral coefficients (MFCCs)**: 13 coefficients were extracted, capturing the overall shape of the spectral envelope.
2. **Spectral Centroid**: Represents the "center of mass" of the spectrum, indicating the brightness of a sound.
3. **Zero-crossing rate**: Measures the number of times the signal changes sign, which is useful for distinguishing percussive sounds.
4. **Chroma Features**: Represents the tonal content of the audio, which can be particularly useful for music classification.

These features were chosen for their proven effectiveness in various audio classification tasks and their ability to capture different aspects of audio signals.

## 4. Model Architecture

Given the small dataset, we opted for classical machine learning models known for their performance on small datasets. We compared three different models:

1. **Random Forest Classifier**: An ensemble learning method that constructs multiple decision trees and outputs the class that is the mode of the classes of the individual trees.
2. **Support Vector Machine (SVM)**: A powerful algorithm that finds the hyperplane that best separates the classes in a high-dimensional space.
3. **K-Nearest Neighbors (KNN)**: A simple yet effective algorithm that classifies a sample based on the majority class of its k nearest neighbors in the feature space.

For each model, we used default hyperparameters to avoid overfitting that could result from extensive tuning on such a small dataset.

## 5. Evaluation Method

To make the most of our limited data, we employed Leave-One-Out Cross Validation (LOOCV). This method:

1. Trains the model on all but one sample
2. Tests the model on the left-out sample
3. Repeats this process for each sample in the dataset

LOOCV provides a more reliable estimate of model performance with small datasets compared to traditional k-fold cross-validation, as it maximizes the amount of data used for training in each iteration.

## 6. Results

Due to the small dataset, specific numerical results may not be reliably indicative of the model's true performance. However, the evaluation process provides insights into which model performs best on our limited data.

The models were compared based on two metrics:

1. **Accuracy**: The proportion of correct predictions among the total number of cases examined.
2. **F1 Score**: The harmonic mean of precision and recall, providing a balanced measure of the model's performance.

The model with the highest F1 score was selected as the final model for deployment.
