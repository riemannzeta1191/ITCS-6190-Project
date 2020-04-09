from flask import Flask
from pyspark.mllib.recommendation import MatrixFactorizationModel
import os
from pyspark import SparkConf, SparkContext
import json
sc = SparkContext(appName="flask")

app = Flask(__name__)
model_path = os.path.join('./models_test', 'movie_lens_als')
same_model = MatrixFactorizationModel.load(sc, model_path)

moviedf = sc.textFile("ml-latest-small/movies.csv")
moviedfh = moviedf.take(1)[0]
small_movies_data = moviedf.filter(lambda line: line!=moviedfh)
#     .map(lambda line: line.split(",")).map(lambda tokens: [tokens[0],tokens[1]]).cache()
# small_movies_titles = small_movies_data.map(lambda x: (int(x[0]),x[1]))

@app.route('/recom/', methods = ['GET', 'POST', 'DELETE'])
def recom_user():

	titles = small_movies_data.map(lambda line: line.split("\|")).map(lambda array: str(array[0])).map(lambda v:v.split(',')).map(lambda k:[k[0],k[1]]).collectAsMap()

	rel = []
	recom = same_model.recommendProducts(20,10)
	for elem in recom:
		rel.append([titles[str(elem.product)],elem.rating])
	print(rel)
	return json.dumps(rel)

@app.route('/recom/all/', methods = ['GET', 'POST', 'DELETE'])
def recommallusers():
	recom = same_model.recommendProductsForUsers(num=20)
	recom1 = recom.collect()
	return json.dumps(recom1)

if __name__ == '__main__':
	print("ll")
	app.run(debug=True, port=5000)