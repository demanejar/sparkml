import os
import shutil

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from flask import Flask, request
from pyspark.ml.clustering import LDA
from pyspark.ml.clustering import LocalLDAModel
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel

import constants
from preprocessing import preprocessing

# init Spark Context
conf = SparkConf().setAppName("Spark ML").setMaster("local[2]")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

# create count vectorizer
cv = CountVectorizer(inputCol="content", outputCol="features", vocabSize=500, minDF=3.0)

# init flask app
app = Flask(__name__)

@app.route("/api/predict")
def predict():
    document = request.args.get("document")
    countVectorizerModel = CountVectorizerModel.load(constants.OUTPUT_PATH + "/Model_CountVectorizer")
    ldaModel = LocalLDAModel.load(constants.OUTPUT_PATH + "/Model_LDA")

    documentDF = sqlContext.createDataFrame([(document, )], ["Content"])
    rdd = documentDF.rdd
    tokens = preprocessing(rdd)
    tokens = tokens.zipWithIndex()

    df = sqlContext.createDataFrame(tokens, ["content", "index"])
    vectorizedToken = countVectorizerModel.transform(df)

    result = ldaModel.transform(vectorizedToken)
    result = result.select("topicDistribution")
    result.show(truncate=False)
    pred = result.rdd.first()
    
    return {"predict": find_max_index(pred['topicDistribution'])}

@app.route("/api/train-model")
def train():
    # read raw data
    data = sqlContext.read.format("csv").options(header='true', inferSchema='true').load(os.path.realpath(constants.PATH))
    rdd = data.rdd

    # preprocessing data
    tokens = preprocessing(rdd)
    tokens = tokens.zipWithIndex()

    df = sqlContext.createDataFrame(tokens, ["content", "index"])

    # vector data
    cvModel = cv.fit(df)
    vectorizedToken = cvModel.transform(df)

    # clustering
    lda = LDA(k=constants.NUM_TOPICS, maxIter=constants.MAX_INTER)
    model = lda.fit(vectorizedToken)

    # ll = model.logLikelihood(vectorizedToken)
    # lp = model.logPerplexity(vectorizedToken)
    # print("The lower bound on the log likelihood of the entire corpus: " + str(ll))
    # print("The upper bound on perplexity: " + str(lp))

    # get vocab 
    vocab = cvModel.vocabulary

    topics = model.describeTopics()
    topicsRdd = topics.rdd

    # result 
    result = model.transform(vectorizedToken)
    result.show()

    if(os.path.isdir(constants.OUTPUT_PATH + "/Model_CountVectorizer")):
        shutil.rmtree(constants.OUTPUT_PATH + "/Model_CountVectorizer")

    cvModel.save(constants.OUTPUT_PATH + "/Model_CountVectorizer")

    if(os.path.isdir(constants.OUTPUT_PATH + "/Model_LDA")):
        shutil.rmtree(constants.OUTPUT_PATH + "/Model_LDA")

    model.save(constants.OUTPUT_PATH + "/Model_LDA")

    return {"message": "successfully", "vocab": vocab}


def find_max_index(arr):
    index = 0
    max = 0
    for i in range(0, len(arr)):
        if arr[i] > max: 
            max = arr[i]
            index = i

    return index

if __name__ == "__main__":
    app.run(debug=True)
