import numpy as np
import librosa
import pickle
from keras.models import model_from_json

# ========== 1. Load model dan scaler ==========
with open("CNN_model.json", "r") as json_file:
    model_json = json_file.read()
loaded_model = model_from_json(model_json)
loaded_model.load_weights("best_model1_weights.h5")
print("✅ Model loaded")

with open("scaler2.pickle", "rb") as f:
    scaler = pickle.load(f)

with open("encoder2.pickle", "rb") as f:
    encoder = pickle.load(f)

print(f"Encoder type: {type(encoder)}")


print("✅ Scaler & encoder loaded")
# Ambil daftar label dari OneHotEncoder
labels = encoder.categories_[0]  # array seperti: ['Angry', 'Happy', 'Sad', ...]


# ========== 2. Fitur Ekstraksi ==========
def zcr(data, frame_length=2048, hop_length=512):
    return np.squeeze(librosa.feature.zero_crossing_rate(y=data, frame_length=frame_length, hop_length=hop_length))

def rmse(data, frame_length=2048, hop_length=512):
    return np.squeeze(librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length))

def mfcc(data, sr, n_mfcc=13, flatten=True):
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc)
    return np.ravel(mfccs.T) if flatten else np.squeeze(mfccs.T)

def extract_features(data, sr=22050):
    z = zcr(data)
    r = rmse(data)
    m = mfcc(data, sr)
    return np.hstack((z, r, m))

# ========== 3. Pad atau potong agar panjang 2376 ==========
def pad_features(features, target_len=2376):
    if len(features) < target_len:
        return np.pad(features, (0, target_len - len(features)), mode='constant')
    else:
        return features[:target_len]

# ========== 4. Siapkan input dari audio ==========
def prepare_input(file_path):
    data, sr = librosa.load(file_path, duration=2.5, offset=0.6)
    features = extract_features(data, sr)
    features = pad_features(features, target_len=2376)
    scaled = scaler.transform([features])
    return np.expand_dims(scaled, axis=2)  # shape: (1, 2376, 1)

# ========== 5. Prediksi ==========
def predict_emotion(audio_path):
    input_data = prepare_input(audio_path)
    prediction = loaded_model.predict(input_data)  # hasil shape (1, n_classes)
    predicted_index = np.argmax(prediction, axis=1)[0]  # index kelas tertinggi
    emotion = labels[predicted_index]  # ambil nama label dari OneHotEncoder
    print(f"🎧 Emosi Terdeteksi: {emotion}")
# ========== 6. Tes ==========
predict_emotion("C:/Users/Aldyn/Desktop/testpy/TA/Software Program/Recording (28).wav")
