import os
import urllib
import zipfile
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.recommendation import MatrixFactorizationModel
import math

import numpy as np


sc = SparkContext(appName="MovieRating")
# Disable the verbose logs


ratingdf = sc.textFile("ml-latest-small/ratings.csv")
ratingdfh = ratingdf.take(1)[0]

small_ratings_data = ratingdf.filter(lambda line: line!=ratingdfh)\
    .map(lambda line: line.split(",")).map(lambda t: (t[0],t[1],t[2])).cache()


moviedf = sc.textFile("ml-latest-small/movies.csv")
moviedfh = moviedf.take(1)[0]
small_movies_data = moviedf.filter(lambda line: line!=moviedfh)\
    .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1])).cache()




training_RDD, validation_RDD, test_RDD = small_ratings_data.randomSplit([6, 2, 2], seed=10)
validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))


seed = 10
iterations = 10

rank = 10

min_error = float('inf')
best_rank = -1


model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations)
predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)

error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())

# print(error)

model = ALS.train(training_RDD, 10, seed=seed, iterations=iterations)
predictions = model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
error_test = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())

print('For testing data the RMSE is %s' % (error_test))


model_path = os.path.join('./models', 'movie_lens_als')


model.save(sc, model_path)
same_model = MatrixFactorizationModel.load(sc, model_path)