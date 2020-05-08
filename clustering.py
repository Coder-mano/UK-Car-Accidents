from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
import anomaly

spark = SparkSession.builder.appName("TSVD").getOrCreate()

def kmeans_clustering(data):
    numeric_cols = ["1st_Road_Number", "Number_of_Vehicles", "Number_of_Casualties", "Longitude", "Latitude",
                    "Speed_limit", "Age_of_Driver"]

    #data = spark.read.parquet("data.parquet")
    data = data.select(numeric_cols)
    vecAssembler = VectorAssembler(inputCols=numeric_cols, outputCol="features").transform(data)
    data_features = vecAssembler.select("features")

    # Kmeans model
    KMeans_model = KMeans(featuresCol="features", k=10, maxIter=100, seed=1234)
    model = KMeans_model.fit(data_features)
    clusters = model.transform(data_features)
    clusters.show()

    # Outlier detection
    centers = model.clusterCenters()
    centers = [center.tolist() for center in centers]

    feature_vectors = data.rdd.map(anomaly.row_to_list)
    min_distances = feature_vectors.map(lambda x: anomaly.min_dist_to_centroid(x, centers))
    min_distances_list = min_distances.collect()  # list of values
    #print len(min_distances_list)
    print min_distances_list  # 4263049 vzdialenosti. Najvacsie su outlier
