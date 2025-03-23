import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import os
from collections import Counter

features_mel = np.load("features_mel.npy")
labels_mel = np.load("labels_mel.npy")

def plot_one_per_class(features, labels):
    unique_classes = np.unique(labels)  
    num_classes = len(unique_classes)

    fig, axes = plt.subplots(num_classes, 1, figsize=(6, 2 * num_classes))

    for i, class_label in enumerate(unique_classes):
        index = np.where(labels == class_label)[0][0]  

        ax = axes[i] if num_classes > 1 else axes
        librosa.display.specshow(features[index], sr=22050, hop_length=512, cmap='magma', ax=ax)
        ax.set_title(f"Class: {class_label}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

plot_one_per_class(features_mel, labels_mel)

dataset_path = "/content/drive/MyDrive/Voice of Birds"
class_counts = {}

for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)
    if os.path.isdir(folder_path):  
        num_files = len([f for f in os.listdir(folder_path) if f.endswith('.mp3')])
        class_counts[folder] = num_files

plt.figure(figsize=(12, 6))
plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
plt.xlabel("Bird Species (Classes)")
plt.ylabel("Number of Audio Files")
plt.title("Distribution of Audio Files per Class")
plt.xticks(rotation=45, ha="right")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
