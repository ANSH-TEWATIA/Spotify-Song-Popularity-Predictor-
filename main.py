import streamlit as st
import numpy as np
import joblib
import pandas as pd

# 🎯 Load model
model = joblib.load("spotify_model.pkl")

# 🎵 Title
st.set_page_config(page_title="Spotify Predictor", layout="centered")
st.title("Spotify Song Popularity Predictor 🎧")

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
genre_columns = [col for col in model.feature_names_in_ if col.startswith("genre_")]

genre = st.selectbox(
    "Select Genre",
    [col.replace("genre_", "") for col in genre_columns]
)

# 🆕 New Features
artist_popularity = st.slider("Artist Popularity", 0.0, 100.0)
duration_min = st.slider("Song Duration (minutes)", 0.0, 10.0)

# 🎯 Predict Button
if st.button("Predict Popularity"):

    # 🧠 Input Data (same as training)
    input_data = {
        'danceability': danceability,
        'energy': energy,
        'loudness': loudness,
        'speechiness': speechiness,
        'acousticness': acousticness,
        'instrumentalness': instrumentalness,
        'liveness': liveness,
        'valence': valence,
        'tempo': tempo,

        'energy_loudness': energy * loudness,
        'dance_energy': danceability * energy,
        'tempo_energy': tempo * energy,

        'artist_popularity': artist_popularity,
        'duration_min': duration_min
    }
input_df = pd.DataFrame([input_data ])

# Add all genre columns as 0
for col in model.feature_names_in_:
    if col.startswith("genre_"):
        input_df[col] = 0

# Set selected genre = 1
genre_col = f"genre_{genre}"
if genre_col in input_df.columns:
    input_df[genre_col] = 1

input_df = input_df[model.feature_names_in_]


prediction = model.predict(input_df)
prediction = np.expm1(prediction)

st.success(f"🎯 Predicted Popularity: {int(prediction[0])}")