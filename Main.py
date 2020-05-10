import corr_matrix
import prepare_data
import classification
from pyspark.sql import SparkSession
import time
from pyspark.ml.feature import VectorAssembler
#import visualization
import clustering

if __name__ == "__main__":

    start = time.time()

    spark = SparkSession.builder.appName("TSVD").getOrCreate()
    data = prepare_data.loadData(spark)
    #data = data.sample(False, 0.02) #run Forrest run! 400s...
    data, indexedData = prepare_data.preprocess(data)

    # Visualization (sroti dlho)
    # data = prepare_data.preprocess(data, indexed=False)
    # visualization.make_barcharts(data, save_pdf=True)
    # visualization.make_histogram(data)

    # Correlations
    col_names = indexedData.schema.names
    # s Accident_Severity_Binary aby bolo vidno korelacie so Severity
    vector_data_corr = VectorAssembler(inputCols=col_names, outputCol="features").transform(indexedData)
    corr_matrix.correlations(vector_data_corr, only_numeric=False)

    # Clustering
    clustering.kmeans_clustering(vector_data_corr, only_numeric=False)

    # Vector data preparation
    col_names.remove('Accident_Severity_Binary')
    vector_data = VectorAssembler(inputCols=col_names, outputCol="features").transform(indexedData)

    # Train/Test split
    training_data, testing_data = vector_data.randomSplit([0.8, 0.2], seed=1234)
    # Modeling
    classification.decisionTree(training_data,testing_data)
    classification.supportVectorMachine(training_data,testing_data)
    classification.randomForrest(training_data,testing_data)
    classification.gradientBoostedTrees(training_data,testing_data)
    classification.naiveBayes(training_data,testing_data)

    end = time.time()
    print end - start

    spark.stop()


