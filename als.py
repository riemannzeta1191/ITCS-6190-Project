import numpy as np
import  pandas as pd

from pyspark import SparkContext
from pyspark.sql import SparkSession

from pyspark.mllib.recommendation import ALS, \
                            MatrixFactorizationModel, Rating
from pyspark.sql import Row

# sc = SparkSession \
#     .builder \
#     .appName("MovieRecommender") \
#     .config("spark.some.config.option", "some-value") \
#     .getOrCreate()
sc = SparkContext(appName="recommender")

moviedf = sc.textFile("ml-latest-small/movies.csv")
# print(moviedf.collect())

ratingdf =  sc.textFile("ml-latest-small/ratings.csv",)
# print(ratingdf.collect())

# als = ALS.train(ratings=ratingdf,iterations=10,)

import math
ratings = ratingdf.map(lambda l: l.split(',')).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
print(ratings.collect())
# Train 80%, Test 20%
trainData, testData = ratings.randomSplit([0.8,0.2],seed=42)
rank = 10
numIterations = 10
testdata = testData.map(lambda p: (p[0], p[1]))

model = ALS.train(trainData, rank,numIterations)
predictions = model.predictAll(testData).map(lambda r: (((r[0]), (r[1])), (r[2])))
rates_and_preds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
print(predictions)


# https://github.com/evancasey/spark-knn-recommender/blob/master/algorithms/itemBasedRecommender.py