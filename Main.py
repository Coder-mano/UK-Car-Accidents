from pyspark.sql import Row, SparkSession
import prepare_data

if __name__ == "__main__":

    spark = SparkSession.builder.appName("TSVD").getOrCreate()


    data = prepare_data.loadData(spark)

    prepare_data.preprocess(data)

    spark.stop()

