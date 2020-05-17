import prepare_data
import statistics
import visualization
import clustering
import classification

from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.ml.feature import VectorAssembler

if __name__ == "__main__":

    spark = SparkSession.builder.appName("TSVD").getOrCreate()

    # Data preparation

    data = prepare_data.loadData(spark)
    data, indexedData = prepare_data.preprocess(data)

    #Parquet
    #data.write.parquet("../data.parquet")
    #indexedData.write.parquet("../ndata.parquet")
    #indexedData = spark.read.parquet("../ndata.parquet")
    #data = spark.read.parquet("../data.parquet")

    col_names = indexedData.schema.names

    # Statistics
    statistics.basic_statistics(data)
    statistics.covariance(data)
    statistics.infoGainRatioTab(indexedData,spark)
    statistics.correlation(indexedData, col_names, spark)

    # Visualizations
    'visualization.make_barcharts(data, save_pdf=False)'
    #visualization.make_histogram(data)     # Runs on pyspark_dist_explore module

    # Vectorization
    col_names.remove('Accident_Severity_Binary')
    indexedData = VectorAssembler(inputCols=col_names, outputCol="features").transform(indexedData)

    # Clustering
    clustering.clustering(indexedData,spark)

    # Train / Test split
    training_data, testing_data = indexedData.randomSplit([0.8, 0.2], seed=1234)

    # Modeling
    classification.supportVectorMachine(training_data,testing_data)
    classification.decisionTree(training_data, testing_data)
    classification.randomForrest(training_data,testing_data)
    classification.gradientBoostedTrees(training_data,testing_data)
    classification.naiveBayes(training_data,testing_data)

    spark.stop()


