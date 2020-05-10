from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
import pandas as pd


def correlations(data, only_numeric=True):
    print "Computing Correlation Matrix"
    col_names = data.schema.names
    col_names.remove('features')
    data = data.drop("Accident_Index", "Local_Authority_Highway")
    if only_numeric:
        numeric_cols = ["1st_Road_Number", "Number_of_Vehicles", "Number_of_Casualties",
                        "Speed_limit", "Age_of_Driver"]
        data = data.select(numeric_cols)
        assembler = VectorAssembler(inputCols=data.columns, outputCol="features")
        data = assembler.transform(data).select("features")

    # Process Data
    matrix = Correlation.corr(data, "features")

    # Display Corr Matrix
    result = matrix.collect()[0]["pearson({})".format("features")].values
    cor_matrix = pd.DataFrame(result.reshape(-1, len(col_names)), columns=col_names, index=col_names)
    print cor_matrix.to_string()
