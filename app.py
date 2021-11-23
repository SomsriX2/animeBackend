from flask import Flask ,request, jsonify
from flask.wrappers import Request
import pandas as pd
from flask_cors import CORS, cross_origin
import warnings
from sklearn.neighbors import NearestNeighbors


app = Flask(__name__)

warnings.filterwarnings(action="ignore")
df_animes = pd.read_csv("anime.csv", index_col="anime_id")
df_animes.head()
# ------------------------------------------------------------------
# Drop the animes with null values
df_clean_animes = df_animes[df_animes.genre.notna() & df_animes.type.notna()]

# First, split the genre column by comma and expand the list so there is
# a column for each genre. Now we have 13 columns, because the anime with
# most genres tags has 13 tags
genres = df_clean_animes.genre.str.split(", ", expand=True)

# Now we can get the list of unique genres. We "convert" the dataframe into
# a single dimension array and take the unique values
unique_genres = pd.Series(genres.values.ravel('K')).dropna().unique()

# Getting the dummy variables will result in having a lot more columns
# than unique genres
dummies = pd.get_dummies(genres)

# So we sum up the columns with the same genre to have a single column for
# each genre
for genre in unique_genres:
    df_clean_animes["Genre: " + genre] = dummies.loc[:, dummies.columns.str.endswith(genre)].sum(axis=1)
    
# Add the type dummies
type_dummies = pd.get_dummies(df_clean_animes.type, prefix="Type:", prefix_sep=" ")
df_clean_animes = pd.concat([df_clean_animes, type_dummies], axis=1)

df_clean_animes = df_clean_animes.drop(columns=["name", "type", "genre", "episodes", "rating", "members"])
df_clean_animes.head()
#------------------------------------------------------------------
# Helper function to get the features of an anime given its name
def get_features_from_anime_name(name):
    return df_clean_animes.loc[df_animes[df_animes.name == name].index]


# Build and "train" the model
neigh = NearestNeighbors(n_neighbors=15)
neigh.fit(df_clean_animes.values)


@app.post("/search-anime")
@cross_origin()
def SearchAnime():
    global Score
    try:
        title = request.get_json()
        # Get the features of this anime
        item_to_compare = get_features_from_anime_name(title['title'])

        # Get the indices of the most similar items found
        # Note: these are ignoring the dataframe indices and starting from 0
        index = neigh.kneighbors(item_to_compare, return_distance=False)

        # Show the details of the items found
        df_animes_clean = df_animes.drop(columns=["type", "episodes", "members"])
        df_animes_clean.loc[df_animes_clean.index[index][0]] 
        #-------------------------------------------------------
        animes_loc = df_animes_clean.loc[df_animes_clean.index[index][0]] 
        Data = animes_loc.astype({"rating":"float"})
        Score=Data.loc[Data["rating"]  > 7.0]
        Score.sort_values(by='rating', ascending=False)
        animelist = Score.values.tolist()
        return jsonify(animelist)
    except ValueError:
        return {"msg": "Please use the capital letter"}
    

@app.get("/getanime")
def getanime():
    return "getanime"

if __name__ == "main":
    app.run(debug=True)