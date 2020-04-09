import os
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.recommendation import MatrixFactorizationModel
import math

import numpy as np


sc = SparkContext(appName="MovieRating")
#
#
#

ratingdf = sc.textFile("ml-latest-small/ratings.csv")
ratingdfh = ratingdf.take(1)[0]
# print(ratingdfh)

small_ratings_data = ratingdf.filter(lambda line: line!=ratingdfh).map(lambda line: line.split(",")).map(lambda t:[t[0],t[1],t[2]])
# print(small_ratings_data.collect())
#
# #
moviedf = sc.textFile("ml-latest-small/movies.csv")
moviedfh = moviedf.take(1)[0]
small_movies_data = moviedf.filter(lambda line: line!=moviedfh)\
    .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1])).cache()
small_movies_titles = small_movies_data.map(lambda x: (int(x[0]),x[1]))

#
# #
# #
training_RDD, validation_RDD, test_RDD = small_ratings_data.randomSplit([6, 2, 2], seed=10)

validation_for_predict_RDD = validation_RDD.map(lambda x: [x[0], x[1]])
# # print(validation_for_predict_RDD.collect())
test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))
print(test_for_predict_RDD.collect())
#
#
seed = 3
iterations = 10
ranks = range(1,3)
rank = 10
lambda_s = np.linspace(0.01,0.5,10)
errors = [0] * (len(ranks)*len(lambda_s))

best_rank = -1
min_error = float('inf')

# #
for rank in range(1,24):
    for lambda_i in lambda_s:
        model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations)
        predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
        rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)

        error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
        if error < min_error:
                    min_error = error
                    best_rank = rank
                    best_lambda = lambda_i

# print(error)

model_test = ALS.train(training_RDD, best_rank, seed=seed, iterations=iterations)
predictions = model_test.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
error_test = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())






model_path = os.path.join('./models_test', 'movie_lens_als')

model_test.save(sc, model_path)

same_model = MatrixFactorizationModel.load(sc, model_path)

print("recommendations:",same_model)
from pyspark.sql.functions import lit

