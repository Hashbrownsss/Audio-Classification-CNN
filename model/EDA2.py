import os
import random
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Define dataset path
dataset_path = "/content/drive/MyDrive/Voice of Birds"

# Count files per class
class_counts = {}
for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)
    if os.path.isdir(folder_path):
        num_files = len([f for f in os.listdir(folder_path) if f.endswith('.mp3')])
        class_counts[folder] = num_files

# Plot class distribution
plt.figure(figsize=(10, 5))
plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
plt.xlabel("Bird Species (Classes)")
plt.ylabel("Number of Audio Files")
plt.title("Distribution of Audio Files per Class")
plt.xticks(rotation=45, ha="right")
plt.show()

# Pick a random audio file from any class
random_class = random.choice(list(class_counts.keys()))
random_folder = os.path.join(dataset_path, random_class)
random_file = random.choice([f for f in os.listdir(random_folder) if f.endswith('.mp3')])
random_path = os.path.join(random_folder, random_file)

y, sr = librosa.load(random_path, sr=22050)
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

plt.figure(figsize=(6, 4))
librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
plt.colorbar(format='%+2.0f dB')
plt.title(f"Mel Spectrogram - {random_class}")
plt.show()
