from pyspark import SparkContext
from pyspark.sql import Row
from pyspark.sql.types import *
from pyspark.sql.functions import unix_timestamp, to_date, when
from pyspark.sql.functions import when, col, sum
from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as sf
from pyspark.sql.functions import isnan, count, avg, mean
from pyspark.ml.feature import StringIndexer
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


def preprocess(data, indexed=False):
    # time & date correction
    data = data.withColumn("Date", sf.to_date("Date", format="dd/MM/yyyy"))  # string to date
    data = data.withColumn("Date", sf.concat(sf.col('Date'), sf.lit(' '), sf.col('Time')))  # concat
    data = data.withColumn("Date", sf.unix_timestamp("Date", format="yyyy-MM-dd HH:mm"))
    data = data.drop("Time")

    # missing data replace
    col_names = data.schema.names
    data = reduce(lambda df, x: df.withColumn(x, to_none(x)), col_names, data)  # replace "-1" with "None"

    # null Location filtering
    data = data.filter(data["Location_Easting_OSGR"].isNotNull() & data["Location_Northing_OSGR"].isNotNull() & data[
        "Longitude"].isNotNull() & data["Latitude"].isNotNull())

    # NA replacement in Junction_Control => where Junction_Location = 0
    update_col = (when((col('Junction_Control').isNull()) & (col('Junction_Location') == 0), 0)
                  .otherwise(col('Junction_Control')))
    data = data.withColumn('Junction_Control', update_col)

    data = filterNa(data)

    # other values replacement
    meanAge = data.filter(data['Age_of_Driver'].isNotNull()).agg(avg(col('Age_of_Driver'))).first()[0]

    values = {'Weather_Conditions': 9,  'Sex_of_Driver': 3,
              'Age_of_Driver': meanAge, 'Age_Band_of_Driver': age_band_count(meanAge),
              'Pedestrian_Location': 10, 'Pedestrian_Movement': 9, 'Pedestrian_Road_Maintenance_Worker': 2,
              'Junction_Control': 0}

    data = data.fillna(value=values)

    # main attr binarization
    data = data.withColumn("Accident_Severity_Binary", when(data["Accident_Severity"] == 1, 0).otherwise(1))

    nominalColumns = ['Police_Force', 'Day_of_Week', 'Local_Authority_(District)', 'Local_Authority_(Highway)',
                      '1st_Road_Class', 'Road_Type', 'Junction_Detail', 'Junction_Control',
                      'Pedestrian_Crossing-Human_Control',
                      'Pedestrian_Crossing-Physical_Facilities', 'Light_Conditions', 'Road_Surface_Conditions',
                      'Special_Conditions_at_Site',
                      'Carriageway_Hazards', 'Urban_or_Rural_Area', 'Did_Police_Officer_Attend_Scene_of_Accident',
                      'Vehicle_Type',
                      'Towing_and_Articulation', 'Vehicle_Manoeuvre', 'Vehicle_Location-Restricted_Lane',
                      'Junction_Location',
                      'Skidding_and_Overturning', 'Hit_Object_in_Carriageway', 'Vehicle_Leaving_Carriageway',
                      'Hit_Object_off_Carriageway',
                      '1st_Point_of_Impact', 'Sex_of_Driver', 'Casualty_Class', 'Casualty_Severity',
                      'Pedestrian_Location',
                      'Pedestrian_Movement', 'Car_Passenger', 'Bus_or_Coach_Passenger',
                      'Pedestrian_Road_Maintenance_Worker',
                      'Casualty_Type', 'Weather_Conditions', 'Age_Band_of_Driver', 'Accident_Severity']

    data = toNominal(data, nominalColumns)

    # reduce -> random sampling | workaround for StringIndexer OOF
    data = data.sample(False, 0.9999)
    # reduce -> stratified sampling
    #data.sampleBy("Accident_Severity", 0.999)
    if indexed:
        print "Indexujem"
        indexedData = getIndexedDataFrame(data, nominalColumns)
        return data, indexedData
    else:
    # temp tests
    # data.select("Vehicle_Propulsion_Code").distinct().show()
    # print data.filter(data["Accident_Severity"] == 1).count()
    # print data.filter(data["Accident_Severity"] == 0).count()

    # nul test ([tested]commented - time consuming)
    # data.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in data.columns]).show()

        return data


def getIndexedDataFrame(data, columns):

   for column in columns:
        data = StringIndexer(inputCol=column, outputCol=column+"_index").fit(data).transform(data).drop(column).withColumnRenamed(column+"_index",column)

   return data


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
                     'LSOA_of_Accident_Location', 'Journey_Purpose_of_Driver')
    return data


def toNominal(data, columns):

    for column in columns:
        data = data.withColumn(column, data[column].cast(StringType()))

    return data


def age_band_count(age):
    if age < 6:
        a_band = 1
    elif 5 < age < 11:
        a_band = 2
    elif 10 < age < 16:
        a_band = 3
    elif 15 < age < 21:
        a_band = 4
    elif 20 < age < 26:
        a_band = 5
    elif 25 < age < 36:
        a_band = 6
    elif 35 < age < 46:
        a_band = 7
    elif 45 < age < 56:
        a_band = 8
    elif 55 < age < 66:
        a_band = 9
    elif 65 < age < 76:
        a_band = 10
    else:
        a_band = 11
    return a_band


def to_none(col_name):
    return when(col(col_name) != "-1", col(col_name)).otherwise(None)


def number_to_text(data):
    data = data.replace(["1", "2", "3"], ["fatal", "serious", "slight"], "Accident_Severity")

    weather = {"1": "fine", "2": "rain", "3": "snow", "4": "fine-wind", "5": "rain-wind", "6": "snow-wind", "7": "fog",
               "8": "other", "9": "unknown"}
    day = {"1": "sunday", "2": "monday", "3": "tuesday", "4": "wednesday", "5": "thursday", "6": "friday",
           "7": "saturday"}
    road_class1 = {"1": "motorway", "2": "A(M)", "3": "A", "4": "B", "5": "C", "6": "unclassified"}
    road_class2 = {"0": "not-junction", "1": "motorway", "2": "A(M)", "3": "A", "4": "B", "5": "C", "6": "unclassified"}
    road_type = {"1": "roundabout", "2": "one-way-street", "3": "dual-carriageway", "6": "single-carriageway",
                 "7": "slip-road", "9": "unknown", "12": "one-way/slip-road"}
    light = {"1": "daylight", "4": "darkness-lit", "5": "darkness-unlit", "6": "darkness-no-light",
             "7": "darkness-unknown"}
    junction = {"0": "not-junction", "1": "roundabout", "2": "mini-roundabout", "3": "t/staggered-junction",
                "5": "slip-road", "6": "crossroads", "7": "more-than-4-arms", "8": "private-drive", "9": "other"}
    junction_control = {"0": "not-junction", "1": "authorised-person", "2": "auto-traffic-signal", "3": "stop-sign",
                        "4": "give-way/uncontrolled"}
    ped_cross_human = {"0": "none", "1": "school-crossing-patrol", "2": "authorised-person"}
    ped_cross_physical = {"0": "no-facilities", "1": "zebra", "4": "pedestrian-light-controlled-crossing",
                          "5": "pedestrian-light", "7": "footbridge/subway", "8": "central-refuge"}
    road_surface = {"1": "dry", "2": "wet", "3": "snow", "4": "ice", "5": "flood", "6": "oil", "7": "mud"}
    spec_conditions = {"0": "none", "1": "auto-signal-out", "2": "auto-signal-defective", "3": "road-sign",
                       "4": "roadworks", "5": "road-surface-defective", "6": "oil", "7": "mud"}
    hazards = {"0": "none", "1": "vehicle", "2": "object", "3": "accident", "4": "dog",
               "5": "pedestrian", "7": "animal"}
    area = {"1": "urban", "2": "rural", "3": "unallocated"}
    police_attend = {"1": "yes", "2": "no", "3": "no"}
    # motorcycle a goods som spojil dokopy
    vehicle_type = {"1": "pedal-cycle", "2": "motorcycle", "3": "motorcycle", "4": "motorcycle", "5": "motorcycle",
                    "8": "taxi", "9": "car", "10": "minibus", "11": "bus", "16": "horse", "17": "agricultural-vehicle",
                    "18": "tram", "19": "van", "20": "goods", "21": "goods", "22": "scooter", "23": "motorcycle",
                    "90": "other", "97": "motorcycle", "98": "goods"}
    sex = {"1": "male", "2": "female", "3": "not-known"}
    # ak by niekoho napadly lepsie nazvy cca podla toho http://www.fsps.muni.cz/emuni/data/reader/book-19/04.html
    age_band = {"1": "child", "2": "school-child", "3": "teen", "4": "child", "5": "young-adult", "6": "adult",
                "7": "middle-adult", "8": "old-adult", "9": "young-senior", "10": "middle-senior", "11": "old-senior"}
    impact = {"0": "no-impact", "1": "front", "2": "back", "3": "offside", "4": "nearside"}
    hit_off_carriageway = {"0": "none", "1": "sign", "2": "lamp", "3": "pole", "4": "tree", "5": "bus-stop",
                           "6": "central-crash-barrier", "7": "near/offside-barrier", "8": "submerged",
                           "9": "entered-ditch", "10": "other-object", "11": "wall"}
    leaving_carriageway = {"0": "didnt-leave", "1": "nearside", "2": "nearside-rebound", "3": "straight-ahead",
                           "4": "offside-central", "5": "offside-central-rebound", "6": "offside-cross-central",
                           "7": "offside", "8": "offside-rebound"}
    hit_in_carriageway = {"0": "none", "1": "previous-accident", "2": "road-works", "4": "parked-vehicle",
                          "5": "bridge(roof)", "6": "bridge(side)", "7": "bollard-of-refuge", "8": "open-door-of-vehicle",
                          "9": "central-island-of-roundabout", "10": "kerb", "11": "other-object", "12": "any-animal"}
    skidding = {"0": "none", "1": "skidded", "2": "skidded-overturned", "3": "jackknifed", "4": "jackknifed-overturned",
                "5": "overturned"}
    junction_location = {"0": "not-junction", "1": "approaching-junction", "2": "cleared-junction",
                         "3": "leaving-roundabout", "4": "entering-roundabout", "5": "leaving-main-road",
                         "6": "entering-main-road", "7": "entering-from-slip-road", "8": "mid-junction"}
    vehicle_location = {"0": "on-main", "1": 'tram-track', "2": "bus-lane", "3": "busway", "4": "cycleway",
                        "5": "cycleway/footway", "6": "on-lay-by", "7": "entering-lay-by", "8": "leaving-lay-by",
                        "9": "footway", "10": "not-on-carriageway"}
    vehicle_manoeuvre = {"1": "reversing", "2": "parked", "3": "w8-to-go", "4": "slowing/stopping", "5": "moving-off",
                         "6": "u-turn", "7": "turning-left", "8": "w8-to-turn-left", "9": "turning-right",
                         "10": "w8-to-turn-right", "11": "changing-lane-to-left", "12": "changing-lane-to-right",
                         "13": "overtaking-moving-vehicle", "14": "overtaking-static-vehicle",
                         "15": "overtaking-nearside", "16": "going-ahead-left-hand-bend",
                         "17": "going-ahead-right-hand-bend", "18": "going-ahead-other"}
    towing = {"0": "no-tow", "1": "articulated", "2": "multiple-trailer", "3": "caravan", "4": "single-trailer",
              "5": "other-tow"}
    casualty_class = {"1": "driver", "2": "passenger", "3": "pedestrian"}
    casualty_severity = {"1": "fatal", "2": "serious", "3": "slight"}
    ped_location = {"0": "not-pedestrian", "1": "crossing-on-pedestrian-crossing-facility",
                    "2": "crossing-in-zig-zag-approach-lines", "3": "crossing-in-zig-zag-exit-lines",
                    "4": "crossing-elsewhere", "5": "in-carriageway-crossing", "6": "on-footway", "7": "on-refuge",
                    "8": "in-centre-of-carriageway", "9": "in-carriageway-not-crossing", "10": "unknown"}
    ped_movement = {"0": "not-pedestrian", "1": "crossing-from-driver-nearside", "2": "crossing-from-nearside-masked",
                    "3": "crossing-from-driver-offside", "4": "crossing-from-offside-masked",
                    "5": "in-carriageway-stationary-not-crossing", "6": "in-carriageway-stationary-not-crossing-masked",
                    "7": "walking-along-in-carriageway-facing-traffic",
                    "8": "walking-along-in-carriageway-back-to-traffic", "9": "unknown"}
    car_passenger = {"0": "not-passenger", "1": "font-seat", "2": "rear-seat"}
    bus_passenger = {"0": "not-passenger", "1": "boarding", "2": "alighting", "3": "standing-passenger",
                     "4": "seated-passenger"}
    maintenance_worker = {"0": "no", "1": "yes", "2": "unknown"}
    casualty_type = {"0": "pedestran", "1": "cyclist", "2": "motorcycler", "3": "motorcycler", "4": "motorcycler",
                     "5": "motorcycler", "8": "taxi-occupant", "9": "car-occupant", "10": "minibus-occupant",
                     "11": "bus-occupant", "16": "horse-rider", "17": "agricultural-occupant", "18": "tram-occupant",
                     "19": "van-occupant", "20": "goods-occupant", "21": "goods-occupant", "22": "scooter-rider",
                     "23": "motocycler", "90": "other-occupant", "97": "motorcycler", "98": "goods-occupant"}

    data = data.replace(to_replace=weather, subset="Weather_Conditions")
    data = data.replace(to_replace=day, subset="Day_of_Week")
    data = data.replace(to_replace=road_class1, subset="1st_Road_class")
    data = data.replace(to_replace=road_class2, subset="2st_Road_class")
    data = data.replace(to_replace=road_type, subset="Road_Type")
    data = data.replace(to_replace=light, subset="Light_Conditions")
    data = data.replace(to_replace=junction, subset="Junction_Detail")
    data = data.replace(to_replace=junction_control, subset="Junction_Control")
    data = data.replace(to_replace=ped_cross_human, subset="Pedestrian_Crossing-Human_Control")
    data = data.replace(to_replace=ped_cross_physical, subset="Pedestrian_Crossing-Physical_Facilities")
    data = data.replace(to_replace=road_surface, subset="Road_Surface_Conditions")
    data = data.replace(to_replace=spec_conditions, subset="Special_Conditions_at_Site")
    data = data.replace(to_replace=hazards, subset="Carriageway_Hazards")
    data = data.replace(to_replace=area, subset="Urban_or_Rural_Area")
    data = data.replace(to_replace=police_attend, subset="Did_Police_Officer_Attend_Scene_of_Accident")
    data = data.replace(to_replace=vehicle_type, subset="Vehicle_Type")
    data = data.replace(to_replace=sex, subset="Sex_of_Driver")
    data = data.replace(to_replace=age_band, subset="Age_Band_of_Driver")
    data = data.replace(to_replace=impact, subset="1st_Point_of_Impact")
    data = data.replace(to_replace=hit_off_carriageway, subset="Hit_Object_off_Carriageway")
    data = data.replace(to_replace=leaving_carriageway, subset="Vehicle_Leaving_Carriageway")
    data = data.replace(to_replace=hit_in_carriageway, subset="Hit_Object_in_Carriageway")
    data = data.replace(to_replace=skidding, subset="Skidding_and_Overturning")
    data = data.replace(to_replace=junction_location, subset="Junction_Location")
    data = data.replace(to_replace=vehicle_location, subset="Vehicle_Location-Restricted_Lane")
    data = data.replace(to_replace=vehicle_manoeuvre, subset="Vehicle_Manoeuvre")
    data = data.replace(to_replace=towing, subset="Towing_and_Articulation")
    data = data.replace(to_replace=casualty_class, subset="Casualty_Class")
    data = data.replace(to_replace=casualty_severity, subset="Casualty_Severity")
    data = data.replace(to_replace=ped_location, subset="Pedestrian_Location")
    data = data.replace(to_replace=ped_movement, subset="Pedestrian_Movement")
    data = data.replace(to_replace=car_passenger, subset="Car_Passenger")
    data = data.replace(to_replace=bus_passenger, subset="Bus_or_Coach_Passenger")
    data = data.replace(to_replace=maintenance_worker, subset="Pedestrian_Road_Maintenance_Worker")
    data = data.replace(to_replace=casualty_type, subset="Casualty_Type")
    return data

