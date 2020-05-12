import prepare_data
#from pyspark_dist_explore import hist  # module for histograms
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pyspark.sql.functions as sf
#import plotille

colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
          '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
          '#ffffff', '#000000']
cols_to_select = ["Accident_Severity", "Weather_Conditions", "Day_of_Week", "1st_Road_class", "Road_Type",
                  "Light_Conditions", "Junction_Detail", "Junction_Control", "Pedestrian_Crossing-Human_Control",
                  "Pedestrian_Crossing-Physical_Facilities", "Road_Surface_Conditions", "Carriageway_Hazards",
                  "Special_Conditions_at_Site", "Urban_or_Rural_Area", "Vehicle_Type", "Sex_of_Driver",
                  "Did_Police_Officer_Attend_Scene_of_Accident", "Age_Band_of_Driver", "1st_Point_of_Impact",
                  "Hit_Object_off_Carriageway", "Vehicle_Leaving_Carriageway", "Hit_Object_in_Carriageway",
                  "Skidding_and_Overturning", "Junction_Location", "Vehicle_Location-Restricted_Lane",
                  "Vehicle_Manoeuvre", "Towing_and_Articulation", "Casualty_Class", "Casualty_Severity",
                  "Pedestrian_Location", "Pedestrian_Movement", "Car_Passenger", "Bus_or_Coach_Passenger",
                  "Pedestrian_Road_Maintenance_Worker", "Casualty_Type"]


def make_barcharts(data, save_pdf=False):
    print "Making BarCharts"
    data = data.select(cols_to_select)
    data = prepare_data.number_to_text(data)
    attributes = data.schema.names
    if not save_pdf:
        for atr in attributes:
            df = data.groupBy(atr).count()
            df.show()
    else:
        print "Saving pdf as bar_plot.pdf"
        with PdfPages("plots/bar_plot.pdf") as pdf:
            for atr in attributes:
                df = data.groupBy(atr).count()
                df.show()
                pandas_df = df.toPandas()
                plt.figure(figsize=(10, 5))
                plt.bar(pandas_df[atr], pandas_df["count"], color=colors)
                plt.tick_params(axis='x', which='both', pad=10, labelrotation=30, labelsize=8)
                plt.title(atr)
                pdf.savefig(bbox_inches="tight")
                plt.close()
        print "Bar plots finished -> PDF saved as bar_plot.pdf"


# Histograms using 3rd party module
# def make_histogram(data):
#     print "Selecting data for histograms"
#     data = data.select(["Age_of_Driver", "Accident_Severity", "Sex_of_Driver"])
#     # Sex_of_Driver & Age_of_Driver
#     print "Filtering data - Sex_of_Driver"
#     male = data.filter(sf.col("Sex_of_Driver") == "1").select(sf.col("Age_of_Driver").alias("Male"))
#     female = data.filter(sf.col("Sex_of_Driver") == "2").select(sf.col("Age_of_Driver").alias("Female"))
#     fig, axes = plt.subplots()
#     fig.set_size_inches(10, 10)
#     hist(axes, [male, female], bins=20, color=["red", "tan"])
#     axes.set_title("Compare Genders")
#     axes.legend()
#     axes.set_xlabel("Age")
#     plt.savefig("plots/genders.png", bbox_inches="tight")
#     plt.close()
#     # Accident_Severity & Age_of_Driver
#     print "Filtering data - Accident_Severity"
#     fatal = data.filter(sf.col("Accident_Severity") == "1").select(sf.col("Age_of_Driver").alias("Fatal"))
#     serious = data.filter(sf.col("Accident_Severity") == "2").select(sf.col("Age_of_Driver").alias("Serious"))
#     slight = data.filter(sf.col("Accident_Severity") == "3").select(sf.col("Age_of_Driver").alias("Slight"))
#     fig, axes = plt.subplots()
#     fig.set_size_inches(10, 10)
#     hist(axes, [fatal, serious, slight], bins=20, color=["red", "tan", "green"])
#     axes.set_title("Compare Severity")
#     axes.legend()
#     axes.set_xlabel("Age")
#     plt.savefig("plots/severity.png", bbox_inches="tight")
#     plt.close()
#     # Accident_Severity(Fatal) & Age_of_driver
#     fig, axes = plt.subplots()
#     fig.set_size_inches(10, 10)
#     hist(axes, fatal, bins=20, color="red")
#     axes.set_title("Severity=Fatal")
#     axes.legend()
#     axes.set_xlabel("Age")
#     plt.savefig("plots/severity-fatal.png", bbox_inches="tight")
#     plt.close()

# 4fun
# def terminal_friendly_histogram(data):
#     print "Age distribution"
#     age = data.select("Age_of_Driver").rdd.flatMap(lambda x: x).collect()
#     print(plotille.hist(age, bins=20))
