import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LabelEncoder,
)
from sklearn.pipeline import Pipeline
import joblib
import xgboost as xgb
from moviepy.editor import VideoFileClip


def extract_audio_from_video(video_path, audio_path):
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)
    except Exception as e:
        print(f"Error extracting audio: {str(e)}")
        return None
    return audio_path


def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, res_type="kaiser_fast")
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)

        return np.hstack(
            (
                np.mean(mfccs, axis=1),
                np.mean(spectral_centroid),
                np.mean(zero_crossing_rate),
                np.mean(chroma, axis=1),
            )
        )
    except Exception as e:
        print(f"Error extracting features from {file_path}: {str(e)}")
        return None


def create_dataset(audio_dir):
    features = []
    labels = []
    for class_name in os.listdir(audio_dir):
        class_path = os.path.join(audio_dir, class_name)
        if os.path.isdir(class_path):
            for audio_file in os.listdir(class_path):
                if not audio_file.endswith(".wav"):
                    continue
                file_path = os.path.join(class_path, audio_file)
                feature = extract_features(file_path)
                if feature is not None:
                    features.append(feature)
                    labels.append(class_name)
    return np.array(features), np.array(labels)


def find_best_model(X_train, y_train):
    models = {
        "RandomForest": RandomForestClassifier(),
        "DecisionTree": DecisionTreeClassifier(),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier(),
        "XGBoost": xgb.XGBClassifier(),
    }

    scalers = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler(),
    }

    param_grids = {
        "RandomForest": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
        "DecisionTree": {"max_depth": [None, 10, 20], "min_samples_split": [2, 5, 10]},
        "SVM": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
        "KNN": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
        "XGBoost": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.3],
        },
    }

    best_score = 0
    best_model = None
    best_scaler = None

    for model_name, model in models.items():
        for scaler_name, scaler in scalers.items():
            pipeline = Pipeline([("scaler", scaler), ("model", model)])

            param_grid = {
                "model__" + key: value for key, value in param_grids[model_name].items()
            }

            grid_search = GridSearchCV(
                pipeline, param_grid, cv=5, scoring="f1_weighted"
            )
            grid_search.fit(X_train, y_train)

            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_model = grid_search.best_estimator_.named_steps["model"]
                best_scaler = grid_search.best_estimator_.named_steps["scaler"]
                print(f"New best model: {model_name} with {scaler_name}")
                print(f"Best parameters: {grid_search.best_params_}")
                print(f"Best F1-score: {best_score:.4f}")
                print()

    return best_model, best_scaler


def train_model(features, labels):
    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels_encoded, test_size=0.2, random_state=42
    )

    best_model, best_scaler = find_best_model(X_train, y_train)

    X_train_scaled = best_scaler.fit_transform(X_train)
    X_test_scaled = best_scaler.transform(X_test)

    best_model.fit(X_train_scaled, y_train)

    y_pred = best_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Best Model Accuracy: {accuracy:.4f}")
    print(f"Best Model F1-score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    return best_model, best_scaler, le


def classify_audio(model, scaler, le, file_path):
    features = extract_features(file_path)
    if features is None:
        return None
    features_scaled = scaler.transform(features.reshape(1, -1))
    prediction = model.predict(features_scaled)[0]
    return le.inverse_transform([prediction])[0]


def save_model():
    audio_dir = "./datasets"
    features, labels = create_dataset(audio_dir)

    best_model, best_scaler, le = train_model(features, labels)

    # Save the best model, scaler, and label encoder for future use
    joblib.dump(best_model, "best_audio_classification_model.joblib")
    joblib.dump(best_scaler, "best_audio_classification_scaler.joblib")
    joblib.dump(le, "label_encoder.joblib")

    print("\nModel, scaler, and label encoder saved successfully.")


def load_model():
    best_model = joblib.load("best_audio_classification_model.joblib")
    best_scaler = joblib.load("best_audio_classification_scaler.joblib")
    le = joblib.load("label_encoder.joblib")

    return best_model, best_scaler, le


def test_model(audio_dir):
    model, scaler, le = load_model()
    for class_name in os.listdir(audio_dir):
        class_path = os.path.join(audio_dir, class_name)
        if os.path.isdir(class_path):
            for audio_file in os.listdir(class_path):
                if not audio_file.endswith(".wav"):
                    continue
                file_path = os.path.join(class_path, audio_file)
                prediction = classify_audio(model, scaler, le, file_path)
                print(f"Prediction for {class_name}: {prediction}")


if __name__ == "__main__":
    save_model()
    test_model("./tests")
