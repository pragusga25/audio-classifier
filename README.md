# Audio Classification Project

This project implements an audio classification system using machine learning techniques to categorize audio files into three classes: music, vehicle sounds, and speech.

## Detailed Report

### Approach

Our approach to audio classification involves the following steps:

1. **Feature Extraction**: We use the librosa library to extract relevant features from audio files. These features capture various aspects of the audio signal that are useful for classification.

2. **Model Selection**: We employ a range of traditional machine learning algorithms and use GridSearchCV for hyperparameter tuning to find the best performing model.

3. **Training and Evaluation**: We split our dataset into training and testing sets, train the models, and evaluate their performance using accuracy and F1-score metrics.

4. **Deployment**: The best performing model is saved for future use in classifying new audio files.

### Features

We extract the following features from each audio file:

1. **Mel-frequency Cepstral Coefficients (MFCCs)**: We use 13 MFCCs, which represent the short-term power spectrum of the audio. MFCCs are widely used in speech and music processing tasks.

2. **Spectral Centroid**: This feature represents the "center of mass" of the spectrum. It indicates where the "center" of the sound is located, helping distinguish between brighter and darker sounds.

3. **Zero Crossing Rate**: This feature provides information about the noisiness of the signal. It's particularly useful for distinguishing between voiced and unvoiced speech segments.

4. **Chroma Features**: These features represent the tonal content of the audio, which is especially useful for music classification.

For each of these features, we calculate the mean value across the entire audio file to create a fixed-length feature vector.

### Model Architecture

We experiment with several traditional machine learning models:

1. **Random Forest**: An ensemble learning method that constructs multiple decision trees and merges them to get a more accurate and stable prediction.

2. **Decision Tree**: A simple yet interpretable model that makes decisions based on asking a series of questions about the features.

3. **Support Vector Machine (SVM)**: A powerful algorithm that finds the hyperplane that best separates the classes in high-dimensional space.

4. **K-Nearest Neighbors (KNN)**: A simple, non-parametric method that classifies a sample based on the majority class of its k nearest neighbors in the feature space.

5. **XGBoost**: An optimized distributed gradient boosting library, known for its speed and performance.

We use GridSearchCV to perform an exhaustive search over specified parameter values for each model. This helps us find the best hyperparameters for each model type.

Additionally, we experiment with different scalers (StandardScaler, MinMaxScaler, and RobustScaler) to normalize our feature set, as some models (like SVM and KNN) are sensitive to the scale of input features.

### Results

After training and evaluating our models, we achieved the following results:

```
Best model: RandomForest with MinMaxScaler
Best parameters: {'model__max_depth': None, 'model__n_estimators': 100}
Best F1-score: 0.5817

Best Model Accuracy: 0.6000
Best Model F1-score: 0.6000

Classification Report:
              precision    recall  f1-score   support

       music       0.00      0.00      0.00         2
      speech       0.80      0.80      0.80         5
     vehicle       0.67      0.67      0.67         3

    accuracy                           0.60        10
   macro avg       0.49      0.49      0.49        10
weighted avg       0.60      0.60      0.60        10
```

#### Interpretation of Results

1. **Overall Performance**: The model achieved an accuracy and F1-score of 0.6000 (60%), which indicates moderate performance. There's significant room for improvement.

2. **Best Model**: The Random Forest classifier with MinMaxScaler performed the best among the tested models. The optimal parameters were 100 trees with no maximum depth limit.

3. **Class-wise Performance**:

   - Speech: The model performed best on speech classification, with precision and recall of 0.80.
   - Vehicle: The model showed moderate performance in classifying vehicle sounds, with precision and recall of 0.67.
   - Music: The model failed to correctly classify any music samples in the test set.

4. **Dataset Imbalance**: The support values (2 for music, 5 for speech, 3 for vehicle) suggest that our test set is small and imbalanced, which could contribute to the poor performance, especially for the music class.

## Setup Instructions

Follow these steps to set up and run the audio classification system:

### 1. Create a virtual environment (optional but recommended)

```sh
python -m venv audio_env
source audio_env/bin/activate  # On Windows, use: audio_env\Scripts\activate
```

### 2. Install required packages

```sh
pip install -r requirements.txt
```

### 3. Install yt-dlp

```sh
pip install yt-dlp
```

### 4. Prepare the dataset

Navigate to the `datasets` folder. Inside, you'll find three subfolders: `music`, `vehicle`, and `speech`. Each subfolder contains a `urls.txt` file with YouTube video URLs.

For each subfolder, run the following command to download audio in WAV format:

```sh
cd ./datasets/music
yt-dlp -x --audio-format wav -a urls.txt
cd ./datasets/vehicle
yt-dlp -x --audio-format wav -a urls.txt
cd ./datasets/speech
yt-dlp -x --audio-format wav -a urls.txt
```

### 5. Prepare the test dataset

Similarly, prepare the test dataset in the `tests` folder:

```sh
cd ./tests/music
yt-dlp -x --audio-format wav -a urls.txt
cd ./tests/vehicle
yt-dlp -x --audio-format wav -a urls.txt
cd ./tests/speech
yt-dlp -x --audio-format wav -a urls.txt
```

### 6. Run the program

Navigate back to the project root directory and run:

```sh
python main.py
```

This will train the model on the dataset, save the best model, and provide classification results for the test files.

## Project Structure

```
audio-classifier/
│
├── datasets/
│   ├── music/
│   │   └── urls.txt
│   ├── vehicle/
│   │   └── urls.txt
│   └── speech/
│       └── urls.txt
│
├── tests/
│   ├── music/
│   │   └── urls.txt
│   ├── vehicle/
│   │   └── urls.txt
│   └── speech/
│       └── urls.txt
│
├── main.py
├── requirements.txt
└── README.md
```
