from pyspark.ml.classification import DecisionTreeClassifier, DecisionTreeClassificationModel, RandomForestClassifier, NaiveBayes, GBTClassifier
from pyspark.ml.classification import LinearSVC
from pyspark.sql.functions import isnan, count, avg, mean
from pyspark.sql import Row


def decisionTree(training_data, testing_data):

    tree_classifier = DecisionTreeClassifier(
        featuresCol="features",
        labelCol="Accident_Severity_Binary",
        impurity="gini",
        maxBins=420,
        maxDepth=5)

    tree_model = tree_classifier.fit(training_data)
    predictions = tree_model.transform(testing_data)

    classificationTest(predictions,testing_data,"Decision Tree")

def supportVectorMachine(training_data, testing_data):

    svm_classifier = LinearSVC(
        featuresCol="features",
        labelCol="Accident_Severity_Binary")

    svm_model = svm_classifier.fit(training_data)
    predictions = svm_model.transform(testing_data)

    classificationTest(predictions,testing_data,"Support Vector Machine")


def naiveBayes(training_data, testing_data):

    nb_classifier = NaiveBayes(
        featuresCol="features",
        labelCol="Accident_Severity_Binary")

    nb_model = nb_classifier.fit(training_data)
    predictions = nb_model.transform(testing_data)

    classificationTest(predictions,testing_data,"Naive Bayes")


def randomForrest(training_data, testing_data):

    rf_classifier = RandomForestClassifier(
        featuresCol="features",
        maxBins=420,
        labelCol="Accident_Severity_Binary")

    rf_model = rf_classifier.fit(training_data)
    predictions = rf_model.transform(testing_data)

    classificationTest(predictions,testing_data,"Random Forrest")


def gradientBoostedTrees(training_data, testing_data):

    gbtc_classifier = GBTClassifier(
        featuresCol="features",
        maxBins=420,
        labelCol="Accident_Severity_Binary")

    gbtc_model = gbtc_classifier.fit(training_data)
    predictions = gbtc_model.transform(testing_data)

    classificationTest(predictions,testing_data,"Gradient Boosted Trees")




def classificationTest(predictions, testing_data, model):

     print "\n" + model
     predictions.stat.crosstab("prediction", "Accident_Severity_Binary").show()
     matrix = predictions.stat.crosstab("prediction", "Accident_Severity_Binary").collect()

     TP = matrix[1]['0.0']
     FP = matrix[1]['1.0']
     TN = matrix[0]['1.0']
     FN = matrix[0]['0.0']

     test_error = (FP + FN) / float(testing_data.count())

     print "Testing error: {0:.4f}".format(test_error)
