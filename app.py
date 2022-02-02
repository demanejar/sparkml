from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from flask import Flask, request
from pyspark.ml.clustering import LocalLDAModel
from pyspark.ml.feature import CountVectorizerModel

import constants
from preprocessing import preprocessing

# init Spark Context
conf = SparkConf().setAppName("Spark ML").setMaster("local[2]")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

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
