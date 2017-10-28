import numpy as np
import pandas as pd
from scipy import sparse
import copy


def get_ratings_data():
    ratings_contents = pd.read_table("data/u.data",
                                     names=["user_id", "movie_id", "rating", "timestamp"])
    return ratings_contents

def get_data_list(filename):
    """
    Extracts the data in a sparse form. Results in a list of length number of users.
    Each element is again a list of tuples (movie_idx, rating)
    :param filename: filename of the dataset under consideration
    :return:
    """
    ratings_contents = pd.read_table(filename,
                                     names=["user", "movie", "rating", "timestamp"])
    highest_user_id = ratings_contents.user.max()
    highest_movie_id = ratings_contents.movie.max()
    ratings = [[] for _ in range(highest_user_id)]
    for _, row in ratings_contents.iterrows():
        # subtract 1 from id's due to match 0 indexing
        ratings[row.user-1].append((row.movie-1, row.rating))
    return ratings

def get_items_data():
    item_contents = pd.read_table("data/u.item",
                                     names=["movie_id", "movie_title", "release_date", "video_release_date",
              "IMDb URL", "unknown", "Action", "Adventure", "Animation",
              "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy" ,
              "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
              "Thriller", "War", "Western"], sep='|', encoding='latin-1')

    return item_contents

def get_users_data():
    user_contents = pd.read_table("data/u.user",
                                     names=["user_id", "age", "gender", "occupation", "zip"], sep='|')

    return user_contents

def make_random_data_like(data):
    ## Please ignore this function !!!
    out_data = [[] for _ in range(len(data))]
    for i,row in enumerate(data):
        for rating in row:
            out_data[i].append((rating[0], int(np.random.randint(1,6,1))))
    return out_data


def calc_rmse(data1, data2):
    """
    Calculates the Root Mean Square Error between the two datasets
    :param data1: first dataset
    :param data2: second dataset
    :return:
    """
    num_users = len(data1)

    SE = 0 #the accumulated Squared Error
    num_total = 0 #the accumulated number of ratings evaluated
    for i in range(num_users):
        data1_dict = dict(data1[i])
        for movie, rating2 in data2[i]:
            #Make one of the datasets into a dictionary to make the search more efficient
            rating1 = data1_dict.get(movie, -1)
            SE += (rating1-rating2)**2
            num_total += 1

            if rating1 == -1:
                print('Could not find rating for movie %i at user %i in data1'%(movie, i))
    rmse = np.sqrt(SE/num_total)
    return rmse

## Load the data
ratings_data = get_ratings_data()
train_data = get_data_list("data/u_train.data")
val_data = get_data_list("data/u_val.data")
test_data = get_data_list("data/u_test.data")

#Ignore the next line, we just need some random data to show the rmse calculation
val_random = make_random_data_like(val_data)

#Example calculation of the rmse
rmse = calc_rmse(val_data, val_random)
print('The RMSE is %5.3f  (for random data, this should be around 2)'%rmse)


items_data = get_items_data()
user_data = get_users_data()

user_ratings = pd.merge(user_data, ratings_data)
all = pd.merge(user_ratings, items_data)

all_matrix = all.values

