import numpy as np
import pandas as pd
from scipy import sparse
import copy
import math


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
    n_users = ratings_contents.user.max()
    n_movies = ratings_contents.movie.max()
    ratings = [[] for _ in range(n_users)]
    for _, row in ratings_contents.iterrows():
        # subtract 1 from id's due to match 0 indexing
        ratings[row.user-1].append((row.movie-1, row.rating))
    return ratings, n_users, n_movies

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

def matrix_factorization(ratings_mat, n_users, n_movies, n_features=15, learn_rate=0.005,
                         regularization_param=0.1,
                         optimizer_pct_improvement_criterion=2):
    user_mat = np.random.rand(n_users, n_features)
    movie_mat = np.random.rand(n_movies, n_features)

    n_total_rated = sum(map(lambda x: len(x), ratings_mat))

    optimizer_iteration_count = 0
    sse_accum = 1E-9
    print("Optimizaiton Statistics")
    print("Iterations | Root Mean Squared Error  |  Percent Improvement")

    while (optimizer_iteration_count < 2 or (pct_improvement > optimizer_pct_improvement_criterion)):
        old_sse = sse_accum
        sse_accum = 0
        for user_idx in range(n_users):
            for movie_idx,rating in ratings_mat[user_idx]:
                    error = rating - np.inner(user_mat[user_idx], movie_mat[movie_idx])
                    sse_accum += error**2
                    user_mat[user_idx] = user_mat[user_idx] + learn_rate * (2 * error * movie_mat[movie_idx] - regularization_param * user_mat[user_idx])
                    movie_mat[movie_idx] = movie_mat[movie_idx] + learn_rate * (2 * error * user_mat[user_idx] - regularization_param * movie_mat[movie_idx])
        pct_improvement = 100 * (old_sse - sse_accum) / old_sse
        print("%d \t\t %f \t\t %f" % (
            optimizer_iteration_count, np.sqrt(sse_accum / n_total_rated), pct_improvement))
        optimizer_iteration_count += 1
    # ensure these are matrices so multiplication works as intended
    return user_mat, movie_mat


def pred_on_test(mat_user, mat_movie, test_data):
    predictions = [[] for _ in range(len(test_data))]
    for user_idx, movies_per_user in enumerate(test_data):
        for movie_idx, _ in movies_per_user:
            pred_rating = int(np.round(np.inner(mat_user[user_idx], mat_movie[movie_idx])))
            predictions[user_idx].append((movie_idx, pred_rating))
    return predictions


## Load the data
# ratings_data = get_ratings_data()
train_data, n_users, n_movies = get_data_list("data/u_train.data")
val_data, _, _ = get_data_list("data/u_val.data")
test_data, _, _ = get_data_list("data/u_test.data")
test_data_label, _, _ = get_data_list("data/u_test_label.data")


#Ignore the next line, we just need some random data to show the rmse calculation
val_random = make_random_data_like(val_data)

#Example calculation of the rmse
rmse = calc_rmse(val_data, val_random)
print('\n For random data, the rmse is %5.3f'%rmse)

#Factorize the matrix
print('Start on factorization')
mat_user, mat_movie = matrix_factorization(train_data, n_users, n_movies)

pred_test = pred_on_test(mat_user, mat_movie, test_data)
rmse_test = calc_rmse(pred_test, test_data_label)
print('With matrix factorization, the rmse is %5.3f'%rmse_test)




