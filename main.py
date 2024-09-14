import os
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from moviepy.editor import VideoFileClip
import joblib
from sklearn.model_selection import LeaveOneOut


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
        print(f"Error extracting features: {str(e)}")
        return None


def create_dataset(audio_dir):
    features = []
    labels = []
    for class_name in os.listdir(audio_dir):
        class_path = os.path.join(audio_dir, class_name)
        if os.path.isdir(class_path):
            for audio_file in os.listdir(class_path):
                file_path = os.path.join(class_path, audio_file)
                feature = extract_features(file_path)
                if feature is not None:
                    features.append(feature)
                    labels.append(class_name)
    return np.array(features), np.array(labels)


def evaluate_model(model, X, y, scaler):
    loo = LeaveOneOut()
    y_true, y_pred = [], []

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train)
        prediction = model.predict(X_test_scaled)

        y_true.append(y_test[0])
        y_pred.append(prediction[0])

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")

    return accuracy, f1


def find_best_model(X, y):
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel="rbf", random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=3),
    }

    scaler = StandardScaler()
    best_score = 0
    best_model = None

    for name, model in models.items():
        accuracy, f1 = evaluate_model(model, X, y, scaler)
        print(f"{name} - Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}")

        if f1 > best_score:
            best_score = f1
            best_model = model

    return best_model, scaler


def train_model(features, labels):
    best_model, scaler = find_best_model(features, labels)

    X_scaled = scaler.fit_transform(features)
    best_model.fit(X_scaled, labels)

    return best_model, scaler


def classify_audio(model, scaler, file_path):
    features = extract_features(file_path)
    if features is None:
        return None
    features_scaled = scaler.transform(features.reshape(1, -1))
    prediction = model.predict(features_scaled)
    return prediction[0]


def generate_model():
    audio_dir = "./audio"
    features, labels = create_dataset(audio_dir)

    best_model, best_scaler = train_model(features, labels)

    # Save the best model and scaler for future use
    joblib.dump(best_model, "best_audio_classification_model.joblib")
    joblib.dump(best_scaler, "best_audio_classification_scaler.joblib")


def test_sample():
    # Example classification
    best_model = joblib.load("best_audio_classification_model.joblib")
    best_scaler = joblib.load("best_audio_classification_scaler.joblib")
    new_audio_file = "./audio.wav"
    result = classify_audio(best_model, best_scaler, new_audio_file)
    print(f"Classification result: {result}")


def run():
    video_file = "./video3.mp4"
    audio_file = "extracted_audio.wav"
    extracted_audio = extract_audio_from_video(video_file, audio_file)
    best_model = joblib.load("best_audio_classification_model.joblib")
    best_scaler = joblib.load("best_audio_classification_scaler.joblib")
    if extracted_audio:
        result = classify_audio(best_model, best_scaler, extracted_audio)
        print(f"Video audio classification result: {result}")


if __name__ == "__main__":
    run()
