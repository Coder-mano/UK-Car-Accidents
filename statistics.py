import prepare_data
from pyspark.sql.functions import concat
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

numeric_attributes = ["Location_Easting_OSGR", "Age_of_Driver", "Location_Northing_OSGR", "Longitude",
                      "Latitude", "Number_of_Vehicles", "Number_of_Casualties", "1st_Road_Number", "Speed_limit"]

# print basics statistics fo number attributes in data
# stats - count, mean, stddev, min, max
# possible to save .csv file
def basic_statistics(data, save_to_file=False):
    numeric_data = data.select(numeric_attributes)
    stats_df = numeric_data.describe(numeric_attributes)
    # get one column stats
    # stats_df = stats_df.select(concat(*stats_df.columns).alias('data'))

    print "\nBasic Statistics:\n"
    stats_df.show()

    if save_to_file:
        print "Saving basic_statistics.csv ..."
        stats_df.write.mode('overwrite').option("header", "true").\
            save("stats/basic_statistics.csv", format="csv")
        print "File basic_statistics.csv saved"


# print covariance between selected numeric attributes of data
def cov_statistics(data, save_to_file=False):
    numeric_data = data.select(numeric_attributes)
    print('\nCovariation between selected attributes:\n')

    print("Age_of_Driver - Longitude")
    print(numeric_data.stat.cov("Age_of_Driver", "Longitude"))

    print("Age_of_Driver - Speed_limit")
    print(numeric_data.stat.cov("Age_of_Driver", "Speed_limit"))

    print("Number_of_Vehicles - Speed_limit")
    print(numeric_data.stat.cov("Number_of_Vehicles", "Speed_limit"))

    print("Number_of_Casualties - Number_of_Vehicles")
    print(numeric_data.stat.cov("Number_of_Casualties", "Number_of_Vehicles"))

    # universal example of covariance between 2 attributes
    # print("Attribute 1 - Attribute 2")
    # print(numeric_data.stat.cov(Attribute 1, Attribute 2))


# print correlation between selected numeric attributes of data
def corr_statistics(data, save_to_file=False):
    numeric_data = data.select(numeric_attributes)
    print('\nCorrelation between selected attributes:\n')

    print("Age_of_Driver - Longitude")
    print(numeric_data.stat.corr("Age_of_Driver", "Longitude"))

    print("Age_of_Driver - Speed_limit")
    print(numeric_data.stat.corr("Age_of_Driver", "Speed_limit"))

    print("Number_of_Vehicles - Speed_limit")
    print(numeric_data.stat.corr("Number_of_Vehicles", "Speed_limit"))

    print("Number_of_Casualties - Number_of_Vehicles")
    print(numeric_data.stat.corr("Number_of_Casualties", "Number_of_Vehicles"))

    # universal example of correlation between 2 attributes
    # print("Attribute 1 - Attribute 2")
    # print(numeric_data.stat.corr(Attribute 1, Attribute 2))


def corr_matrix_pearson(spark, data, save_to_file=False):
    numeric_data = data.select(numeric_attributes)

    print "\n Correlation Matrix - Pearson Method: \n"
    vector_col = "corr_attributes"
    assembler = VectorAssembler(inputCols=numeric_attributes, outputCol=vector_col)
    data_vector = assembler.transform(numeric_data).select(vector_col)

    matrix = Correlation.corr(data_vector, vector_col).collect()[0][0]
    matrix_values = matrix.toArray().tolist()
    matrix_data_frame = spark.createDataFrame(matrix_values, numeric_attributes)
    matrix_data_frame.show()

    if save_to_file:
        # save values of matrix to .csv file
        print "Saving corr_matrix_pearson.csv ..."
        matrix_data_frame.write.mode('overwrite').option("header", "true").\
            save("stats/corr_matrix_pearson.csv", format="csv")
        print "File corr_matrix_pearson.csv saved"