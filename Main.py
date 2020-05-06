import prepare_data
from pyspark.sql import SparkSession
import time

if __name__ == "__main__":

    start = time.time()
    spark = SparkSession.builder.appName("TSVD").getOrCreate()
    data = prepare_data.loadData(spark)
    data = prepare_data.preprocess(data)
    prepare_data.number_to_text(data).show()
    end = time.time()
    print end - start

    spark.stop()

