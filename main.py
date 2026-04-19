import streamlit as st
import numpy as np
import joblib

# 🎯 Load model
model = joblib.load("spotify_model.pkl")

# 🎵 Title
st.title("Spotify Song Popularity Predictor 🎧")
st.set_page_config(page_title="Spotify Predictor", layout="centered")

st.markdown("### Enter song features below:")

# 🎚️ Basic Features
danceability = st.slider("Danceability", 0.0, 1.0)
energy = st.slider("Energy", 0.0, 1.0)
loudness = st.slider("Loudness", -60.0, 0.0)
speechiness = st.slider("Speechiness", 0.0, 1.0)
acousticness = st.slider("Acousticness", 0.0, 1.0)
instrumentalness = st.slider("Instrumentalness", 0.0, 1.0)
liveness = st.slider("Liveness", 0.0, 1.0)
valence = st.slider("Valence", 0.0, 1.0)
tempo = st.slider("Tempo", 50.0, 200.0)

# 🆕 New Features
artist_popularity = st.slider("Artist Popularity", 0.0, 100.0)
duration_min = st.slider("Song Duration (minutes)", 0.0, 10.0)

# 🎯 Predict Button
if st.button("Predict Popularity"):

    # 🧠 Input Data (same as training)
    input_data = np.array([[
        danceability,
        energy,
        loudness,
        speechiness,
        acousticness,
        instrumentalness,
        liveness,
        valence,
        tempo,

        energy * loudness,
        danceability * energy,
        tempo * energy,

        artist_popularity,
        duration_min
    ]])

    # 🔮 Prediction
    prediction = model.predict(input_data)

    # 🔥 Reverse log transform
    prediction = np.expm1(prediction)

    st.success(f"🎯 Predicted Popularity: {int(prediction[0])}")