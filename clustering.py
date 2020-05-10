from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
import anomaly

numeric_cols = ["Number_of_Vehicles", "Number_of_Casualties", "Speed_limit", "Age_of_Driver"]


def kmeans_clustering(data, only_numeric=False):
    print "K-Means Clustering"

    if only_numeric:
        data = data.select(numeric_cols)
        data = VectorAssembler(inputCols=numeric_cols, outputCol="features").transform(data)

    data_features = data.select("features")
    # Kmeans model
    kmeans_model = KMeans(featuresCol="features", k=10, maxIter=100, seed=1234)
    model = kmeans_model.fit(data_features)
    clusters = model.transform(data_features)
    clusters.show()

    # Outlier detection
    centers = model.clusterCenters()
    centers = [center.tolist() for center in centers]

    print "Computing distances"
    feature_vectors = data.rdd.map(anomaly.row_to_list)
    min_distances = feature_vectors.map(lambda x: anomaly.min_dist_to_centroid(x, centers))
    min_distances_list = min_distances.collect()  # list of values
    #print len(min_distances_list)  # 4263049 vzdialenosti.
    #print max(min_distances_list)  # Najvacsie su outlier

    print "Creating pandas DF"
    data = data.toPandas()
    print "Adding column to DF"
    data["distances"] = min_distances_list
    print "Finding largest distances"
    largest = data.nlargest(10, ['distances'])
    print largest.to_string()
