"""
Odtial som to ukradol:
https://github.com/amida-tech/spark-anomaly/blob/master/app_dataframe.py
"""

from pyspark.sql.types import IntegerType
from scipy.spatial import distance
import math


def cast_columns(spark_df, column_list):
    for column in column_list:
        spark_df = spark_df.withColumn(column, spark_df[column].cast(IntegerType()))
    return spark_df


def dist_to_centroid(point_vector, model, k):
    distances = []
    for cluster_number in range(k):
        centroid = model.clusterCenters()[cluster_number]
        dist = distance.euclidean(point_vector,centroid)
        distances.append(dist)
        return(distances)


def min_dist_to_centroid(point_vector, centers):
    distances = []
    for center_point in centers:
        dist = calculate_euclidean_distance(point_vector, center_point)
        distances.append(dist)
    min_distance = min(distances)
    return min_distance


def row_to_list(row):
    columns = row.__fields__
    list_format = []
    for column in columns:
        list_format.append(row[column])
    return list_format


def calculate_euclidean_distance(vector1, vector2):
    dist = [(a - b)**2 for a, b in zip(vector1, vector2)]
    dist = math.sqrt(sum(dist))
    return dist
