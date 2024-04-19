import streamlit as st
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

def main():
    st.title('Emotion Recognition from Audio')
    
    # Load the pre-trained model
    model = load_model()
    
    file_to_be_uploaded = st.file_uploader("Upload an audio file (.wav)", type="wav")

    if file_to_be_uploaded:
        st.audio(file_to_be_uploaded, format='audio/wav')
        predicted_emotion = predict_emotion(model, file_to_be_uploaded)
        st.success(f'Predicted emotion: {predicted_emotion}')


if __name__ == "__main__":
    main()
