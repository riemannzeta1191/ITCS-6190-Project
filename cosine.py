from surprise import Dataset
from surprise import KNNBasic
import numpy as np
from surprise.model_selection import cross_validate

data = Dataset.load_builtin('ml-100k')

trainingSet = data.build_full_trainset()

sim_options = {
	'name': 'cosine',
	'user_based': False
}


def calccosine(utility, u1, u2):
	movies = {}

	for movie in utility[u1]:
		if movie in utility[u2]:
			movies[movie] = 1
	length = len(movies)

	if length == 0:
		return 0

	sum_xy = sum_xx = sum_yy = 0

	for movie in movies:
		sum_xx += pow(utility[u1][movie], 2)
		sum_yy += pow(utility[u2][movie], 2)
		sum_xy += (utility[u1][movie] * utility[u2][movie])

	numerator = sum_xy
	denominator = pow(sum_xx * sum_yy, 0.5)

	if denominator == 0:
		return 0
	else:
		return numerator / denominator


knn = KNNBasic(sim_options=sim_options)
knn.fit(trainingSet)

testSet = trainingSet.build_anti_testset()
predictions = knn.test(testSet)

from collections import defaultdict


def get_top3_recommendations(predictions, topN=10):
	top_recs = defaultdict(list)
	for uid, iid, true_r, est, _ in predictions:
		top_recs[uid].append((iid, est))

	for uid, user_ratings in top_recs.items():
		user_ratings.sort(key=lambda x: x[1], reverse=True)
		top_recs[uid] = user_ratings[:topN]

	return top_recs


import os, io


def read_item_names():


	file_name = (os.path.expanduser('~') +
				 '/.surprise_data/ml-100k/ml-100k/u.item')
	rid_to_name = {}
	with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
		for line in f:
			line = line.split('|')
			rid_to_name[line[0]] = line[1]

	return rid_to_name

top3_recommendations = get_top3_recommendations(predictions)
rid_to_name = read_item_names()
for uid, user_ratings in top3_recommendations.items():
	print(uid, [rid_to_name[iid] for (iid, _) in user_ratings])
