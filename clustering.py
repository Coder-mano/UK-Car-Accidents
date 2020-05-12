from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
import numpy as np
numeric_cols = ["Number_of_Vehicles", "Number_of_Casualties", "Speed_limit", "Age_of_Driver"]


def euclidean_distance(point_vector, centers):
    distances = []
    for center_point in centers:
        pv = np.array(point_vector)
        cp = np.array(center_point)
        dist = np.linalg.norm(pv - cp)
        distances.append(dist)
    min_distance = min(distances)  # select minimal distance
    return min_distance


def row_to_list(row):
    columns = row.__fields__
    list_format = []
    for column in columns:
        list_format.append(row[column])
    return list_format


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
    feature_vectors = data_features.rdd.map(row_to_list)
    distances = feature_vectors.map(lambda x: euclidean_distance(x, centers))
    distances_list = distances.collect()  # list of values
    converted_list = [float(elm) for elm in distances_list]
    return converted_list

    # #  pretty print
    # print "Creating pandas DF"
    # data = data.toPandas()
    # print "Adding column to DF"
    # data["distances"] = min_distances_list
    # print "Finding largest distances"
    # largest = data.nlargest(10, ['distances'])
    # print largest.to_string()
