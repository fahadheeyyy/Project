import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# def plot_mfcc(file_path, save_path):
#     y, sr = librosa.load(file_path, sr=None)
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

#     plt.figure(figsize=(10, 4))
#     librosa.display.specshow(mfcc, sr=sr, x_axis='time')
#     plt.colorbar(label='MFCC Coefficients')
#     plt.title('MFCC Features')
#     plt.xlabel('Time')
#     plt.ylabel('MFCC Coefficients')
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()

# # Example Usage
# plot_mfcc("OAF_bath_sad.wav", "mfcc_plot.png")

# def plot_zcr(file_path, save_path):
#     y, sr = librosa.load(file_path, sr=None)
#     zcr = librosa.feature.zero_crossing_rate(y)

#     plt.figure(figsize=(10, 4))
#     plt.plot(zcr[0])
#     plt.title('Zero Crossing Rate')
#     plt.xlabel('Frames')
#     plt.ylabel('Rate')
#     plt.grid()
#     plt.savefig(save_path)
#     plt.close()

# # Example Usage
# plot_zcr("OAF_bath_sad.wav", "zcr_plot.png")

# def plot_rms(file_path, save_path):
#     y, sr = librosa.load(file_path, sr=None)
#     rms = librosa.feature.rms(y=y)

#     plt.figure(figsize=(10, 4))
#     plt.plot(rms[0])
#     plt.title('Root Mean Square (RMS) Energy')
#     plt.xlabel('Frames')
#     plt.ylabel('Energy')
#     plt.grid()
#     plt.savefig(save_path)
#     plt.close()

# # Example Usage
# plot_rms("OAF_bath_sad.wav", "rms_plot.png")

# def plot_chroma(file_path, save_path):
#     y, sr = librosa.load(file_path, sr=None)
#     chroma = librosa.feature.chroma_stft(y=y, sr=sr)

#     plt.figure(figsize=(10, 4))
#     librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma')
#     plt.colorbar(label='Chroma Intensity')
#     plt.title('Chroma STFT')
#     plt.xlabel('Time')
#     plt.ylabel('Pitch Class')
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()

# # Example Usage
# plot_chroma("OAF_bath_sad.wav", "chroma_plot.png")

# def plot_mel_spectrogram(file_path, save_path):
#     y, sr = librosa.load(file_path, sr=None)
#     mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
#     mel_db = librosa.power_to_db(mel_spec, ref=np.max)

#     plt.figure(figsize=(10, 4))
#     librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel')
#     plt.colorbar(label='dB')
#     plt.title('Mel Spectrogram')
#     plt.xlabel('Time')
#     plt.ylabel('Mel Frequency')
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()

# # Example Usage
# plot_mel_spectrogram("OAF_bath_sad.wav", "mel_spectrogram.png")

import tensorflow as tf
from tensorflow.keras.utils import plot_model
import visualkeras
from tensorflow.keras import layers, models
from PIL import ImageFont  # For better font rendering (optional)

# Define the CNN model
def create_model():
    model = models.Sequential([
        layers.Conv1D(16, 3, activation='relu', input_shape=(13, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        
        layers.Conv1D(32, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Create model
model = create_model()

# Save standard model plot
plot_model(model, to_file="cnn_model.png", show_shapes=True, show_layer_names=True)
print("CNN Model architecture saved as 'cnn_model.png'.")

# Visualize using visualkeras
try:
    font = ImageFont.truetype("arial.ttf", 20)  # Load font (optional)
except:
    font = None  # If font is unavailable, default will be used

visualkeras.layered_view(model, to_file="cnn_architecture.png", legend=True, font=font)
print("Visualized CNN architecture saved as 'cnn_architecture.png'.")



