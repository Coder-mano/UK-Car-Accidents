from pyspark import SparkContext
from pyspark.sql import Row
from pyspark.sql.types import *
from pyspark.sql.functions import unix_timestamp, to_date, when
from pyspark.sql.functions import when, col , sum
from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as sf
from pyspark.sql.functions import isnan, count, avg, mean
import numpy
import csv

def loadData(spark):

    # CSV loading
    df_accidents = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("./raw_data/Accidents.csv")
    df_casualties = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("./raw_data/Casualties.csv")
    df_vehicles = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("./raw_data/Vehicles.csv")

    df_casualties = df_casualties.withColumnRenamed("Vehicle_Reference", "Vehicle_Reference_Casualty")


    data = df_accidents.join(df_casualties, "Accident_Index").drop(df_casualties.Accident_Index)
    data = data.join(df_vehicles, "Accident_Index").drop(df_vehicles.Accident_Index)

    return data




def preprocess(data):

    # time & date correction
    data = data.withColumn("Date", sf.to_date("Date", format="dd/MM/yyyy"))  # string to date
    data = data.withColumn("Date", sf.concat(sf.col('Date'), sf.lit(' '), sf.col('Time')))  # concat
    data = data.withColumn("Date", sf.unix_timestamp("Date", format="yyyy-MM-dd HH:mm"))
    data = data.drop("Time")

    # missing data replace
    col_names = data.schema.names
    data = reduce(lambda df, x: df.withColumn(x, to_none(x)), col_names, data)  # replace "-1" with "None"

    # null Location filtering
    data = data.filter(data["Location_Easting_OSGR"].isNotNull() & data["Location_Northing_OSGR"].isNotNull() & data["Longitude"].isNotNull() & data["Latitude"].isNotNull() )

    # NA replacement in Junction_Control => where Junction_Location = 0
    update_col = (when((col('Junction_Control').isNull()) & (col('Junction_Location') == 0), 0)
                  .otherwise(col('Junction_Control')))
    data = data.withColumn('Junction_Control', update_col)

    data = filterNa(data)

    # other values replacement
    meanAge = data.filter(data['Age_of_Driver'].isNotNull()).agg(avg(col('Age_of_Driver'))).first()[0]

    values = {'Weather_Conditions': 9, 'Journey_Purpose_of_Driver': 15, 'Sex_of_Driver': 3,
              'Age_of_Driver': meanAge,'Age_Band_of_Driver': age_band_count(meanAge),
              'Pedestrian_Location': 10, 'Pedestrian_Movement': 9, 'Pedestrian_Road_Maintenance_Worker': 2,
              'Junction_Control': 0}

    data = data.fillna(value=values)

    # main attr binarization
    data = data.withColumn("Accident_Severity", when(data["Accident_Severity"] == 1, 0).otherwise(1))

    # :TODO takto by to dakto mohol spravit ja nestiham ;)
    #data = data.replace('6', 'snow', 'wheather_condition')
    data = data.replace('0', 'Fatal', 'Accident_Severity')
    data = data.replace('1', 'NonFatal', 'Accident_Severity')

    # fast way
    data = toNominal(data)

    # 'nominal' test
    data.printSchema()


    # temp tests
    #data.select("Vehicle_Propulsion_Code").distinct().show()
    #print data.filter(data["Accident_Severity"] == 1).count()
    #print data.filter(data["Accident_Severity"] == 0).count()

    # nul test ([tested]commented - time consuming)
    #data.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in data.columns]).show()

    print data.count()


def filterNa(data):
    # Na row filtering
    data = data.dropna(subset=['Location_Easting_OSGR', 'Location_Northing_OSGR', 'Longitude', 'Latitude',
                               '1st_Road_Number', 'Junction_Detail', 'Pedestrian_Crossing-Human_Control',
                               'Pedestrian_Crossing-Physical_Facilities', 'Road_Surface_Conditions',
                               'Special_Conditions_at_Site', 'Carriageway_Hazards', 'Vehicle_Type',
                               'Towing_and_Articulation',
                               'Vehicle_Manoeuvre', 'Vehicle_Location-Restricted_Lane', 'Skidding_and_Overturning',
                               'Hit_Object_in_Carriageway', 'Vehicle_Leaving_Carriageway',
                               'Hit_Object_off_Carriageway', '1st_Point_of_Impact', 'Car_Passenger',
                               'Bus_or_Coach_Passenger', 'Date', 'Junction_Location',
                               'Did_Police_Officer_Attend_Scene_of_Accident'])

    # NA Columns deletion
    data = data.drop('2nd_Road_Number', '2nd_Road_Class',
                     'Was_Vehicle_Left_Hand_Drive?', 'Engine_Capacity_(CC)',
                     'Propulsion_Code', 'Age_of_Vehicle', 'Driver_IMD_Decile',
                     'Driver_Home_Area_Type',
                     'Sex_of_Casualty', 'Age_of_Casualty', 'Age_Band_of_Casualty', 'Casualty_Home_Area_Type',
                     'LSOA_of_Accident_Location')
    return data



def toNominal(data):
    data = data.withColumn('Police_Force', data['Police_Force'].cast(StringType()))
    data = data.withColumn('Day_of_Week', data['Day_of_Week'].cast(StringType()))
    data = data.withColumn('Local_Authority_(District)', data['Local_Authority_(District)'].cast(StringType()))
    data = data.withColumn('Local_Authority_(Highway)', data['Local_Authority_(Highway)'].cast(StringType()))
    data = data.withColumn('1st_Road_Class', data['1st_Road_Class'].cast(StringType()))
    data = data.withColumn('Road_Type', data['Road_Type'].cast(StringType()))
    data = data.withColumn('Junction_Detail', data['Junction_Detail'].cast(StringType()))
    data = data.withColumn('Junction_Control', data['Junction_Control'].cast(StringType()))
    data = data.withColumn('Pedestrian_Crossing-Human_Control',
                           data['Pedestrian_Crossing-Human_Control'].cast(StringType()))
    data = data.withColumn('Pedestrian_Crossing-Physical_Facilities',
                           data['Pedestrian_Crossing-Physical_Facilities'].cast(StringType()))
    data = data.withColumn('Light_Conditions', data['Light_Conditions'].cast(StringType()))
    data = data.withColumn('Road_Surface_Conditions', data['Road_Surface_Conditions'].cast(StringType()))
    data = data.withColumn('Special_Conditions_at_Site', data['Special_Conditions_at_Site'].cast(StringType()))
    data = data.withColumn('Carriageway_Hazards', data['Carriageway_Hazards'].cast(StringType()))
    data = data.withColumn('Urban_or_Rural_Area', data['Urban_or_Rural_Area'].cast(StringType()))
    data = data.withColumn('Did_Police_Officer_Attend_Scene_of_Accident',
                           data['Did_Police_Officer_Attend_Scene_of_Accident'].cast(StringType()))
    data = data.withColumn('Vehicle_Type', data['Vehicle_Type'].cast(StringType()))
    data = data.withColumn('Towing_and_Articulation', data['Towing_and_Articulation'].cast(StringType()))
    data = data.withColumn('Vehicle_Manoeuvre', data['Vehicle_Manoeuvre'].cast(StringType()))
    data = data.withColumn('Vehicle_Location-Restricted_Lane',
                           data['Vehicle_Location-Restricted_Lane'].cast(StringType()))
    data = data.withColumn('Junction_Location', data['Junction_Location'].cast(StringType()))
    data = data.withColumn('Skidding_and_Overturning', data['Skidding_and_Overturning'].cast(StringType()))
    data = data.withColumn('Hit_Object_in_Carriageway', data['Hit_Object_in_Carriageway'].cast(StringType()))
    data = data.withColumn('Vehicle_Leaving_Carriageway', data['Vehicle_Leaving_Carriageway'].cast(StringType()))
    data = data.withColumn('Hit_Object_off_Carriageway', data['Hit_Object_off_Carriageway'].cast(StringType()))
    data = data.withColumn('1st_Point_of_Impact', data['1st_Point_of_Impact'].cast(StringType()))
    data = data.withColumn('Journey_Purpose_of_Driver', data['Journey_Purpose_of_Driver'].cast(StringType()))
    data = data.withColumn('Sex_of_Driver', data['Sex_of_Driver'].cast(StringType()))
    data = data.withColumn('Casualty_Class', data['Casualty_Class'].cast(StringType()))
    data = data.withColumn('Casualty_Severity', data['Casualty_Severity'].cast(StringType()))
    data = data.withColumn('Pedestrian_Location', data['Pedestrian_Location'].cast(StringType()))
    data = data.withColumn('Pedestrian_Movement', data['Pedestrian_Movement'].cast(StringType()))
    data = data.withColumn('Car_Passenger', data['Car_Passenger'].cast(StringType()))
    data = data.withColumn('Bus_or_Coach_Passenger', data['Bus_or_Coach_Passenger'].cast(StringType()))
    data = data.withColumn('Pedestrian_Road_Maintenance_Worker',
                           data['Pedestrian_Road_Maintenance_Worker'].cast(StringType()))
    data = data.withColumn('Casualty_Type', data['Casualty_Type'].cast(StringType()))
    data = data.withColumn('Junction_Location', data['Junction_Location'].cast(StringType()))
    data = data.withColumn('Junction_Location', data['Junction_Location'].cast(StringType()))
    data = data.withColumn('Junction_Location', data['Junction_Location'].cast(StringType()))
    data = data.withColumn('Junction_Location', data['Junction_Location'].cast(StringType()))
    data = data.withColumn('Junction_Location', data['Junction_Location'].cast(StringType()))
    data = data.withColumn('Junction_Location', data['Junction_Location'].cast(StringType()))
    data = data.withColumn('Junction_Location', data['Junction_Location'].cast(StringType()))
    data = data.withColumn('Junction_Location', data['Junction_Location'].cast(StringType()))
    data = data.withColumn('Junction_Location', data['Junction_Location'].cast(StringType()))
    return data


def age_band_count(age):
    if age < 6:
        a_band = 1
    elif age > 5 and age < 11:
        a_band = 2
    elif age >10 and age < 16:
        a_band = 3
    elif age > 15 and age < 21:
        a_band = 4
    elif age > 20 and age < 26:
        a_band = 5
    elif age > 25 and age < 36:
        a_band = 6
    elif age > 35 and age < 46:
        a_band = 7
    elif age > 45 and age < 56:
        a_band = 8
    elif age > 55 and age < 66:
        a_band = 9
    elif age > 65 and age < 76:
        a_band = 10
    else:
        a_band = 11
    return a_band

def to_none(col_name):
    return when(col(col_name) != "-1", col(col_name)).otherwise(None)
