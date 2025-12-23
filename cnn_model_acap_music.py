import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, GlobalAveragePooling2D, Rescaling
from keras.optimizers import Adam
from keras.metrics import Precision, Recall
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import os  # Add this import
from sklearn.metrics import confusion_matrix
import pandas as pd
os.makedirs('/Users/balamuralibalu/PythonProjects/voice_controlla/Models', exist_ok=True)


file_paths = [
    '/Users/balamuralibalu/PythonProjects/voice_controlla/npz_folder/acap_data_batch_1.npz',
    '/Users/balamuralibalu/PythonProjects/voice_controlla/npz_folder/fma_data_batch_1.npz',
]

spectrograms_list = []
labels_list = []

for file in file_paths:
    data = np.load(file)
    spectrograms_list.append(data['spectrograms'])
    labels_list.append(data['labels'])

# Concatenate along first axis (number of samples)
X_s_db = np.concatenate(spectrograms_list, axis=0)
y = np.concatenate(labels_list, axis=0)

print("Combined X shape:", X_s_db.shape)
print("Combined y shape:", y.shape)
# Check class distribution (should be ~1000 each for each batch of 1000)
num_ones = np.sum(y)
print("Number of ones in y:", num_ones)
num_zeros = y.size - num_ones
print("Number of zeros in y:", num_zeros)

# Example for dataset-level standardization
mean = np.mean(X_s_db) # Dataset-wide mean across all pixels
std = np.std(X_s_db)   # Dataset-wide std across all pixels
print("TRAIN_MEAN:", mean)
print("TRAIN_STD:", std)

X_norm = (X_s_db - mean) / std  # Centers at 0, scales to unit variance
X_norm = X_norm[..., np.newaxis]  # add channel dim for CNN (samples, height, Width, channels for RGB). Here channels=1 (monochromatic)
X = X_norm
# Use the arrays as needed
print("Dimensions to feed into CNN:", X.shape, y.shape)

#Train test split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=42,stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state = 42,stratify=y_train)
print("Train:", np.bincount(y_train))
print("Val:  ", np.bincount(y_val))
print("Test: ", np.bincount(y_test))

#Build model
input_shape = (2049, 1723, 1)

model = Sequential([
    Input(shape=(2049, 1723, 1)),
    
    # Aggressive downsampling
    Conv2D(16, (5,5), strides=8, activation='relu'),  # →256x215
    MaxPooling2D((2,2)),
    Dropout(0.3),
    
    Conv2D(32, (3,3), strides=4, activation='relu'),  # →64x53
    MaxPooling2D((2,2)),
    Dropout(0.4),
    
    Conv2D(64, (3,3), activation='relu'),             # →32x26
    GlobalAveragePooling2D(),  # NO FLATTEN!
    
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


#Compile model
optimizer = Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', Precision(), Recall()]
)
model.summary()

#fit
history = model.fit(X_train, y_train, batch_size=32, epochs = 5, validation_data=(X_val,y_val), verbose=1)

# Plot training loss and validation loss (bias & variance)
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.legend()
plt.show()

# Optionally, plot training and validation accuracy
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epoch')
plt.legend()
plt.show()

#Evaluate model
loss, accuracy, precision, recall = model.evaluate(X_test, y_test, batch_size=32)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")
print(f"Test precision: {precision}")
print(f"Test recall: {recall}")


# Get predictions

y_prob = model.predict(X_test)
y_pred = (y_prob >= 0.5).astype(int).ravel()
print("\nPer-class metrics:")
print(classification_report(y_test, y_pred, target_names=["Acap", "Music"]))

cm = confusion_matrix(y_test, y_pred)  # order: [0, 1] = [Acap, Music]

cm_df = pd.DataFrame(
    cm,
    index=["True Acap", "True Music"],
    columns=["Pred Acap", "Pred Music"]
)

print("\nConfusion matrix:")
print(cm_df)

#Save model
model.save('/Users/balamuralibalu/PythonProjects/voice_controlla/Models/acapella_vs_music_detector.keras')


# file_paths = [
#     #'/Users/balamuralibalu/PythonProjects/AI_Music_detection_project/AI_music_detection/out/data_batch_1.npz',
#     '/Users/balamuralibalu/PythonProjects/AI_Music_detection_project/AI_music_detection/out/data_batch_2.npz',
#     '/Users/balamuralibalu/PythonProjects/AI_Music_detection_project/AI_music_detection/out/data_batch_3.npz',
#     '/Users/balamuralibalu/PythonProjects/AI_Music_detection_project/AI_music_detection/out/data_batch_4.npz',
# ]

# for i, file in enumerate(file_paths):
#     testing_data = np.load(file)
#     X_testing = testing_data["spectrograms"]
#     y_testing = testing_data["labels"]
#     ones = np.sum(y_testing)
#     print(f"{ones} ones out of 1000")
#     mean = np.mean(X_testing)
#     std = np.std(X_testing)
#     X_testing_norm = (X_testing - mean)/std
#     X_testing_norm = X_testing_norm[...,np.newaxis]



# # Get predictions
# y_prob = model.predict(X_test, batch_size=32)
# y_pred = (y_prob >= 0.5).astype(int).ravel()

# # Print the table you want
# y_prob = model.predict(X_test)
# y_pred = (y_prob >= 0.5).astype(int).ravel()
# print("\nPer-class metrics:")
# print(classification_report(y_test, y_pred, target_names=["Human", "Suno"]))