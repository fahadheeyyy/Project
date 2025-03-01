import tkinter as tk
from tkinter import filedialog, messagebox
import tensorflow as tf
import joblib
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import os
from threading import Thread
import warnings


warnings.simplefilter(action='ignore', category=UserWarning)


# Load model and scaler
def load_model_and_scaler():
    print("Loading model and scaler...")
    model = tf.keras.models.load_model('depression_detection_model.keras')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

# Extract features from audio file
def extract_features_from_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)

# Predict depression probability
def predict_single_sample(features, model, scaler):
    scaled_features = scaler.transform(features.reshape(1, -1))
    scaled_features = scaled_features.reshape(-1, 13, 1)
    prediction = model.predict(scaled_features, verbose=0)
    return prediction[0][0]

# Record audio
def live_prediction(duration=8, sr=44100, save_path=None):
    print("Recording...")
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    print("Recording complete.")
    if save_path:
        sf.write(save_path, recording, sr)
        print(f"Recording saved to {save_path}")
    return recording.flatten(), sr

# Function to run prediction
def run_prediction(model, scaler, audio_file=None, recording=None, duration=None, save_path=None):
    if audio_file:
        example_features = extract_features_from_audio(audio_file)
    elif recording is not None:
        temp_file_path = save_path if save_path else "temp_recording.wav"
        sf.write(temp_file_path, recording, 22050)
        example_features = extract_features_from_audio(temp_file_path)
        # if not save_path:
        #     os.remove(temp_file_path)

    probability = predict_single_sample(example_features, model, scaler)
    return probability

# UI logic
class DepressionDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Depression Detection System")

        self.model, self.scaler = load_model_and_scaler()

        self.root.configure(bg='#f5f5f5')  
        self.label = tk.Label(root, text="Depression Detection", font=("Helvetica", 30, "bold"), fg="#00796b", bg="#f5f5f5")
        self.label.pack(pady=30)

        self.status_label = tk.Label(root, text="", font=("Arial", 14), fg="#757575", bg="#f5f5f5")
        self.status_label.pack(pady=10)

        self.record_button = tk.Button(
            root, text="Record Audio", command=self.start_recording, font=("Arial", 14, "bold"),
            bg="#29b6f6", fg="#ffffff", activebackground="#0288d1", activeforeground="#ffffff",
            relief="flat", width=20
        )
        self.record_button.pack(pady=20)

        self.upload_button = tk.Button(
            root, text="Upload Audio File", command=self.upload_audio, font=("Arial", 14, "bold"),
            bg="#ff7043", fg="#ffffff", activebackground="#d84315", activeforeground="#ffffff",
            relief="flat", width=20
        )
        self.upload_button.pack(pady=20)

        self.result_label = tk.Label(root, text="Prediction: ", font=("Arial", 18), fg="#616161", bg="#f5f5f5")
        self.result_label.pack(pady=30)

        # self.footer_label = tk.Label(root, text="Developed by Fahad", font=("Arial", 10), fg="#757575", bg="#f5f5f5")
        # self.footer_label.pack(side="bottom", pady=10)

    def update_status(self, text, color="#757575"):
        self.status_label.config(text=text, fg=color)

    def start_recording(self):
        self.update_status("Recording in progress...", color="#f57c00")
        duration = 8  

        def record_and_predict():
            try:
                recording, sr = live_prediction(duration)
                self.update_status("Processing the recording...", color="#0288d1")
                probability = run_prediction(self.model, self.scaler, recording=recording, duration=duration)
                self.show_prediction(probability)
                self.update_status("Recording complete!", color="#388e3c")
            except Exception as e:
                self.update_status("Error during recording!", color="#d32f2f")
                messagebox.showerror("Error", str(e))

        Thread(target=record_and_predict).start()

    def upload_audio(self):
        file_path = filedialog.askopenfilename(title="Select an Audio File", filetypes=(("WAV Files", "*.wav"), ("All Files", "*.*")))
        if file_path:
            self.update_status("Processing uploaded audio...", color="#0288d1")
            try:
                probability = run_prediction(self.model, self.scaler, audio_file=file_path)
                self.show_prediction(probability)
                self.update_status("Audio processing complete!", color="#388e3c")
            except Exception as e:
                self.update_status("Error during processing!", color="#d32f2f")
                messagebox.showerror("Error", str(e))

    def show_prediction(self, probability):
        prediction_text = f"Depression Probability: {probability:.2%}\nPrediction: {'Depression' if probability > 0.5 else 'Normal'}"
        self.result_label.config(text=prediction_text, fg="#d32f2f" if probability > 0.5 else "#388e3c")
        print(f"depression probability is :{probability}")



if __name__ == "__main__":
    root = tk.Tk()
    app = DepressionDetectionApp(root)
    root.mainloop()
