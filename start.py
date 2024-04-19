import tkinter as tk
from tkinter import filedialog
import numpy as np    
import tensorflow as tf
import os
import librosa

# Function to load the pre-trained model
def load_model():
    model = tf.keras.models.load_model('mymodel.h5')
    return model

# Function to extract MFCC features from an audio file
def extract_mfcc(wav_file_name):
    y, sr = librosa.load(wav_file_name)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs

# Function to predict the emotion from an audio file
def predict_emotion(model, wav_filepath):
    emotions = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'}
    test_point = extract_mfcc(wav_filepath)
    test_point = np.reshape(test_point, newshape=(1, 40, 1))
    predictions = model.predict(test_point)
    return emotions[np.argmax(predictions[0]) + 1]

def browse_file():
    filename = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if filename:
        label.config(text="Selected file: " + os.path.basename(filename))
        audio_label.config(text="Playing audio...")
        audio_label.after(2000, lambda: audio_label.config(text=""))

        predicted_emotion = predict_emotion(model, filename)
        result_label.config(text="Predicted emotion: " + predicted_emotion)

if __name__ == "__main__":
    model = load_model()

    root = tk.Tk()
    root.title("Emotion Detection Using Audio")

    # Set the size of the window
    root.geometry("400x250")

    heading_label = tk.Label(root, text="Emotion Detection Using Audio", font=("Helvetica", 16))
    heading_label.pack(pady=10)

    label = tk.Label(root, text="")
    label.pack()

    browse_button = tk.Button(root, text="Browse", command=browse_file)
    browse_button.pack()

    audio_label = tk.Label(root, text="")
    audio_label.pack()

    result_label = tk.Label(root, text="")
    result_label.pack()

    root.mainloop()
