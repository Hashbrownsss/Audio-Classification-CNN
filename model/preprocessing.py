import os
import librosa
import numpy as np

# Define dataset path
dataset_path = "/content/drive/MyDrive/Voice of Birds"

sr = 22050
n_mels = 128
fixed_duration = 2.0
n_fft = 2048
hop_length = 512

def load_spectrogram(audio_path):
    try:
        y, _ = librosa.load(audio_path, sr=sr, duration=fixed_duration)
        if len(y) < fixed_duration * sr:  
            y = np.pad(y, (0, int(fixed_duration * sr) - len(y)), mode='constant')

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        return mel_spec_db 

    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None  
    
features_mel = []
labels_mel = []

for root, _, files in os.walk(dataset_path):
    class_label = os.path.basename(root)  
    for file in files:
        if file.endswith(".mp3"):
            audio_path = os.path.join(root, file)
            spectrogram = load_spectrogram(audio_path)

            if spectrogram is not None:
                features_mel.append(spectrogram)
                labels_mel.append(class_label)

features_mel = np.array(features_mel)
labels_mel = np.array(labels_mel)

np.save("features_mel.npy", features_mel)
np.save("labels_mel.npy", labels_mel)

print("Preprocessing Complete!")
print("Feature shape:", features_mel.shape)
print("Labels shape:", labels_mel.shape)
