import numpy as np
import pandas as pd
from scipy import sparse


def get_ratings_data():
    ratings_contents = pd.read_table("data/u.data",
                                     names=["user", "movie", "rating", "timestamp"])
    highest_user_id = ratings_contents.user.max()
    highest_movie_id = ratings_contents.movie.max()
    ratings_as_mat = sparse.lil_matrix((highest_user_id, highest_movie_id))
    for _, row in ratings_contents.iterrows():
        # subtract 1 from id's due to match 0 indexing
        ratings_as_mat[row.user-1, row.movie-1] = row.rating
    return ratings_contents, ratings_as_mat


def matrix_factorization(ratings_mat, n_features=8, learn_rate=0.005,
                         regularization_param=0.02,
                         optimizer_pct_improvement_criterion=2):
    n_users = ratings_mat.shape[0]
    n_movies = ratings_mat.shape[1]
    n_already_rated = ratings_mat.nonzero()[0].size
    user_mat = np.random.rand(
        n_users*n_features).reshape([n_users, n_features])
    movie_mat = np.random.rand(
        n_movies*n_features).reshape([n_features, n_movies])

    optimizer_iteration_count = 0
    sse_accum = 0
    print("Optimizaiton Statistics")
    print("Iterations | Mean Squared Error  |  Percent Improvement")
    while (optimizer_iteration_count < 2 or (pct_improvement > optimizer_pct_improvement_criterion)):
        old_sse = sse_accum
        sse_accum = 0
        for i in range(n_users):
            for j in range(n_movies):
                if ratings_mat[i, j] > 0:
                    error = ratings_mat[i, j] - \
                            np.dot(user_mat[i, :], movie_mat[:, j])
                    sse_accum += error**2
                    for k in range(n_features):
                        user_mat[i, k] = user_mat[
                                             i, k] + learn_rate * (2 * error * movie_mat[k, j] - regularization_param * user_mat[i, k])
                        movie_mat[k, j] = movie_mat[
                                              k, j] + learn_rate * (2 * error * user_mat[i, k] - regularization_param * movie_mat[k, j])
        pct_improvement = 100 * (old_sse - sse_accum) / old_sse
        print("%d \t\t %f \t\t %f" % (
            optimizer_iteration_count, sse_accum / n_already_rated, pct_improvement))
        old_sse = sse_accum
        optimizer_iteration_count += 1
    # ensure these are matrices so multiplication works as intended
    return np.matrix(user_mat), np.matrix(movie_mat)


def pred_one_user(user_mat, movie_mat, user_id):
    return user_mat[user_id] * movie_mat


ratings_data_contents, ratings_mat = get_ratings_data()
user_mat, movie_mat = matrix_factorization(ratings_mat)
print(pred_one_user(user_mat, movie_mat, 1))
