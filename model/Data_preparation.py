import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

features_mel = np.load("features_mel.npy")
labels_mel = np.load("labels_mel.npy")

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels_mel)
labels_onehot = to_categorical(labels_encoded, num_classes=len(np.unique(labels_encoded)))

features_mel = features_mel[..., np.newaxis]  

x_train, x_temp, y_train, y_temp = train_test_split(features_mel, labels_onehot, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

np.save("x_train.npy", x_train)
np.save("y_train.npy", y_train)
np.save("x_val.npy", x_val)
np.save("y_val.npy", y_val)
np.save("x_test.npy", x_test)
np.save("y_test.npy", y_test)

np.save("label_encoder.npy", label_encoder.classes_)

print("Data Preparation Complete!")
print("Train shape:", x_train.shape, y_train.shape)
print("Validation shape:", x_val.shape, y_val.shape)
print("Test shape:", x_test.shape, y_test.shape)
