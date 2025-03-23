import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

sr = 22050
n_mels = 128
fixed_duration = 3.0
n_fft = 2048
hop_length = 512

def load_spectrogram(audio_path):
    try:
        y, _ = librosa.load(audio_path, sr=sr, duration=fixed_duration)
        if len(y) < fixed_duration * sr:  #padding if file size is small
            y = np.pad(y, (0, int(fixed_duration * sr) - len(y)), mode='constant')

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length) #make into spec's using STFT
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max) # power scale to decibel

        return mel_spec_db 

    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None 

def plot_mel_spectrogram(spectrogram, title="Mel Spectrogram"):
    plt.figure(figsize=(6, 4))
    librosa.display.specshow(spectrogram, sr=sr, hop_length=hop_length, cmap='magma')
    plt.title(title)
    plt.axis('off')
    plt.colorbar(format='%+2.0f dB')
    plt.show()

def plot_multiple_spectrograms(features, labels, num_samples=5):
    plt.figure(figsize=(6, num_samples * 3))
    
    for i in range(min(num_samples, len(features))):
        plt.subplot(num_samples, 1, i + 1)
        librosa.display.specshow(features[i], sr=sr, hop_length=hop_length, cmap='magma')
        plt.title(f"Class: {labels[i]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
