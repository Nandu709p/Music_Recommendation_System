import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

# data set
data_path = "./data/SpotifyAudioFeaturesApril2019.csv" 
data = pd.read_csv(data_path)

# feature selection
features = ["danceability", "energy", "valence", "tempo", "acousticness", "instrumentalness", "speechiness"]
df_features = data[features]

#pre-processing
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(df_features)

#  k-NN algorithm
model = NearestNeighbors(n_neighbors=10, metric="euclidean")  
model.fit(normalized_features)

# Function to recommend songs based on input song index
def recommend_songs(song_index, data, n_recommendations=5):
    distances, indices = model.kneighbors([normalized_features[song_index]], n_neighbors=n_recommendations + 1)
    recommendations = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx != song_index: 
            recommendations.append({
                "song": data.iloc[idx]["track_name"],
                "artist": data.iloc[idx]["artist_name"],
                "distance": dist
            })
    return recommendations

song_index = 0  
recommendations = recommend_songs(song_index, data)

print(f"Recommendations for '{data.iloc[song_index]['track_name']}' by {data.iloc[song_index]['artist_name']}:\n")
for rec in recommendations:
    print(f"- {rec['song']} by {rec['artist']} (distance: {rec['distance']:.4f})")

# saving the model
import pickle
with open("knn_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f) 
