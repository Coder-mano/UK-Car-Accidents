from pyspark.ml.classification import DecisionTreeClassifier, DecisionTreeClassificationModel
from pyspark.ml.classification import LinearSVC


def decisionTree(training_data, testing_data):

    tree_classifier = DecisionTreeClassifier(
        featuresCol="features",
        labelCol="Accident_Severity_Binary",
        impurity="gini",
        maxBins=420,
        maxDepth=5)

    tree_model = tree_classifier.fit(training_data)
    predictions = tree_model.transform(testing_data)

    predictions.stat.crosstab("prediction", "Accident_Severity_Binary").show()
    classificationTest(predictions,testing_data)

def supportVectorMachine(training_data, testing_data):

    svm_classifier = LinearSVC(
        featuresCol="features",
        labelCol="Accident_Severity_Binary")

    svm_model = svm_classifier.fit(training_data)
    predictions = svm_model.transform(testing_data)

    predictions.stat.crosstab("prediction", "Accident_Severity_Binary").show()


def naiveBayes(training_data, testing_data):

    nb_classifier = LinearSVC(
        featuresCol="features",
        labelCol="Accident_Severity_Binary")

    nb_model = nb_classifier.fit(training_data)
    predictions = nb_model.transform(testing_data)

    predictions.stat.crosstab("prediction", "Accident_Severity_Binary").show()


def randomForrest(training_data, testing_data):

    rf_classifier = RandomForestClassifier(
        featuresCol="features",
        labelCol="Accident_Severity_Binary")

    rf_model = rf_classifier.fit(training_data)
    predictions = rf_model.transform(testing_data)

    predictions.stat.crosstab("prediction", "Accident_Severity_Binary").show()


def gradientBoostedTrees(training_data, testing_data):

    gbtc_classifier = GBTClassifier(
        featuresCol="features",
        labelCol="Accident_Severity_Binary")

    gbtc_model = gbtc_classifier.fit(training_data)
    predictions = gbtc_model.transform(testing_data)

    predictions.stat.crosstab("prediction", "Accident_Severity_Binary").show()




def classificationTest(predictions, testing_data):
     test_error = predictions.filter(predictions["prediction"] != predictions["Accident_Severity_Binary"])
     print 'Counted' #OOF?
     test_error = test_error.count() / float(testing_data.count())
     print "Testing error: {0:.4f}".format(test_error)
