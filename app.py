from flask import Flask, request, render_template
import pandas as pd
import pickle

# Initialize Flask app
app = Flask(__name__)

#giving paths to model, scaler, and dataset
model_path = "./models/knn_model.pkl"
scaler_path = "./models/scaler.pkl"
data_path = "./data/SpotifyAudioFeaturesApril2019.csv"

# using the pre-trained model and scaler
try:
    with open(model_path, "rb") as f:
        knn_model = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError as e:
    raise FileNotFoundError("Model or scaler file not found. Verify file paths.") from e

# reading dataset
data = pd.read_csv(data_path)
features = ["danceability", "energy", "valence", "tempo", "acousticness", "instrumentalness", "speechiness"]

# Normalizing  the features into a range 0 - 1 
try:
    normalized_features = scaler.transform(data[features])
except KeyError:
    raise KeyError("Ensure the dataset contains all required features: " + ", ".join(features))


def find_song_index(song_name):
    matching_rows = data[data['track_name'].str.lower() == song_name.lower()]
    if not matching_rows.empty:
        return matching_rows.index[0]
    return None


def recommend_songs(song_index, n_recommendations=5):
    distances, indices = knn_model.kneighbors([normalized_features[song_index]], n_neighbors=n_recommendations + 1)
    recommendations = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx != song_index:
            recommendations.append({
                "track_name": data.iloc[idx]["track_name"],
                "artist_name": data.iloc[idx]["artist_name"],
                "distance": round(dist, 4)
            })
    return recommendations

# Route to Home page
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        song_name = request.form.get("song_name")  
        if not song_name:
            return render_template("index.html", error="Please enter a song name.")
        
        try:
            song_index = find_song_index(song_name)
            if song_index is not None:
                recommendations = recommend_songs(song_index)
                return render_template(
                    "index.html",
                    recommendations=recommendations,
                    input_song=song_name
                )
            else:
                return render_template("index.html", error="Song not found! Please try another name.")
        except Exception as e:
            return render_template("index.html", error=f"An error occurred: {str(e)}")
    return render_template("index.html")

# running flask app
if __name__ == "__main__":
    app.run(debug=True)
