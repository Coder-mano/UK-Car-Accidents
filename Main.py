import prepare_data
from pyspark.sql import SparkSession

if __name__ == "__main__":

    spark = SparkSession.builder.appName("TSVD").getOrCreate()

    data = prepare_data.loadData(spark)
    data = prepare_data.preprocess(data)
    data.show()

    spark.stop()

