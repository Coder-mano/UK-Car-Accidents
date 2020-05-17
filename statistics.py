import prepare_data
from pyspark.sql.functions import concat
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from numpy import log2, append, argsort, transpose,array
from pyspark.sql.functions import when, col, sum, count
from pyspark.sql import SQLContext

numeric_attributes = [ "Age_of_Driver", "Number_of_Vehicles", "Number_of_Casualties", "Speed_limit"]

# print basics statistics fo number attributes in data
# stats - count, mean, stddev, min, max
def basic_statistics(data):
    numeric_data = data.select(numeric_attributes)
    stats_df = numeric_data.describe(numeric_attributes)

    print "\nNumeric Statistics:"
    stats_df.show()


def correlation(indexedData, col_names, spark):

    print '\nCorrelations'
    correlations = []
    for firstColumn in col_names:
        subCorrelations = []
        for secondColumn in col_names:
            # Get the sub correlation for each column
            subCorrelations.append('{0: .2f}'.format(indexedData.stat.corr(secondColumn, firstColumn)))
        correlations.append(subCorrelations)

    names = indexedData.schema.names
    correlationMatrix = zip(names, *correlations)
    # Correlation matrix print
    spark.createDataFrame(correlationMatrix,schema= ['Correlations'] + col_names).show(30,truncate = False)


# Numeric attributes covariance
def covariance(data):

    print('Covariance:')

    numeric_data = data.select(numeric_attributes)
    tmp_attributes = numeric_attributes

    for firstColumn in numeric_attributes:
        tmp_attributes = tmp_attributes[1:]

        for secondColumn in tmp_attributes:
            print(firstColumn + ' - ' + secondColumn) + '{0: .2f}'.format(numeric_data.stat.cov(firstColumn, secondColumn))


# Entropy calculation
def calcEntropy(data, attr):
    # Frequency table
    attr_freq = data.groupby(attr).count()
    attr_freq = array(attr_freq.select(attr, 'count').collect())
    total = (float) (data.count())
    # Entropy calculation formula
    entropy = -(attr_freq[:,1]/total * log2(attr_freq[:,1]/total)).sum()
    return entropy


# Information gain ratio calculation
def informationGainRatio(data, main_attr, target_attr, total_count, total_entropy):

    # Frequency table with True and False counts
    attr_freq = data.groupby(target_attr).agg(count(target_attr).alias("count"), count(when(col(main_attr) == 1, 1)).alias("True_count"), count(when((col(main_attr) == 0),0)).alias("False_count"))

    # Order ascending if numerical attribute
    if dict(data.dtypes)[target_attr] == 'int':
        attr_freq = attr_freq.orderBy(target_attr)
    # Frequency table to numpy array (idx: 0-count 1-True_count 2-False_count)
    attr_freq = array(attr_freq.select('count', 'True_count', 'False_count').collect()).astype('float')

    # Information gain ratio for a numerical attribute
    if dict(data.dtypes)[target_attr] == 'int':
        # cumulative sum (idx: 0-count_cumsum  1-True_cumsum   2-False_cumsum)
        cs = attr_freq.cumsum(axis=0)
        cs = cs[:-1, :]
        # Information gain ratio calculation for each threshold
        total_T = attr_freq[:,1].sum()
        total_F = attr_freq[:,2].sum()
        less = -1 * ((cs[:,1]/cs[:,0]) * safe_log2((cs[:,1]/cs[:,0])) + (cs[:,2]/cs[:,0]) * safe_log2((cs[:,2]/cs[:,0])))
        more = -1 * (((total_T-cs[:,1])/(total_count - cs[:,0]))
                     * safe_log2(((total_T-cs[:,1])/(total_count - cs[:,0])))
                     +((total_F - cs[:,2])/(total_count - cs[:,0]))
                     * safe_log2((total_F - cs[:,2])/(total_count - cs[:,0])))
        attr_entropy = less * (cs[:,0]/total_count) + more * ((total_count - cs[:,0])/total_count)
        information_gain = total_entropy - attr_entropy
        # Intrinsic value for each threshold
        intrinsic_val = -1 *((cs[:,0]/total_count)*safe_log2(cs[:,0]/total_count)
                             + ((total_count-cs[:,0])/total_count) * safe_log2(((total_count-cs[:,0])/total_count)))
        information_gain_ratio = information_gain/intrinsic_val
        # Information gain ratio = best value
        information_gain_ratio = max(information_gain_ratio)
    else:
        # Information gain ratio of nominal attribute
        attr_entropy = -1 * ((attr_freq[:,0]/total_count) * ((attr_freq[:,1]/ attr_freq[:,0]) * safe_log2(attr_freq[:,1]/attr_freq[:,0]) + (attr_freq[:,2]/attr_freq[:,0]) * safe_log2(attr_freq[:,2]/attr_freq[:,0])))
        information_gain = total_entropy - attr_entropy.sum()
        intrinsic_val = (-1 * ((attr_freq[:,0]/total_count) * safe_log2(attr_freq[:,0]/total_count)))
        information_gain_ratio = information_gain / intrinsic_val.sum()

    return information_gain_ratio

# log2() undefined handling
def safe_log2(arr):
    arr[arr == 0.0] = 1.0
    return log2(arr)

# Information gain ratio of each attribute
def infoGainRatioTab(data, spark):
    arr = data.schema.names
    total = (float) (data.count())
    # Entropy of space
    total_entropy = calcEntropy(data, 'Accident_Severity_Binary')
    # Calculate information gain ratio for each attribute
    arr2 = map(lambda x: informationGainRatio(data, 'Accident_Severity_Binary', x, total, total_entropy), arr)
    arr2 = [float("{:.4f}".format(round(i, 4))) for i in arr2]
    sqlContext = SQLContext(spark.sparkContext)
    info_gain_ratio_tab = sqlContext.createDataFrame(zip(arr, arr2), schema=['Attributes', 'Information Gain Ratio'])
    info_gain_ratio_tab = info_gain_ratio_tab.orderBy('Information Gain Ratio', ascending=False)
    print '\nInformation Gain Ratio'
    info_gain_ratio_tab.show(30,truncate = False)

    return info_gain_ratio_tab
