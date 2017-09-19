import numpy as np
import pandas as pd

def get_ratings_data():
    ratings_contents = pd.read_table("data/u.data",
                                     names=["user_id", "movie_id", "rating", "timestamp"])
    return ratings_contents

def get_items_data():
    item_contents = pd.read_table("data/u.item",
                                     names=["movie_id", "movie_title", "release_date", "video_release_date",
              "IMDb URL", "unknown", "Action", "Adventure", "Animation",
              "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy" ,
              "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
              "Thriller", "War", "Western"], sep='|')

    return item_contents

def get_users_data():
    user_contents = pd.read_table("data/u.user",
                                     names=["user_id", "age", "gender", "occupation", "zip"], sep='|')

    return user_contents

ratings_data = get_ratings_data()
items_data = get_items_data()
user_data = get_users_data()

user_ratings = pd.merge(user_data, ratings_data)
all = pd.merge(user_ratings, items_data)

all_matrix = all.values

