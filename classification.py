from pyspark.ml.classification import DecisionTreeClassifier, DecisionTreeClassificationModel, RandomForestClassifier, NaiveBayes, GBTClassifier
from pyspark.ml.classification import LinearSVC
from pyspark.sql.functions import isnan, count, avg, mean
from pyspark.sql import Row
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def decisionTree(training_data, testing_data):

    tree_classifier = DecisionTreeClassifier(
        featuresCol="features",
        labelCol="Accident_Severity_Binary",
        impurity="entropy",
        maxBins=100,
        maxDepth=25,
        seed=123)

    tree_model = tree_classifier.fit(training_data)
    predictions = tree_model.transform(testing_data)

    classificationTest(predictions,testing_data,"Decision Tree")


def randomForrest(training_data, testing_data):

    rf_classifier = RandomForestClassifier(
        labelCol="Accident_Severity_Binary",
        featuresCol="features",
        maxBins=100,
        impurity="gini",
        numTrees=25,
        maxDepth=13,
        seed=123)

    rf_model = rf_classifier.fit(training_data)
    predictions = rf_model.transform(testing_data)

    classificationTest(predictions,testing_data,"Random Forrest")


def supportVectorMachine(training_data, testing_data):

    svm_classifier = LinearSVC(
        featuresCol="features",
        labelCol="Accident_Severity_Binary"
    )

    svm_model = svm_classifier.fit(training_data)
    predictions = svm_model.transform(testing_data)

    classificationTest(predictions,testing_data,"Support Vector Machine")


def naiveBayes(training_data, testing_data):

    nb_classifier = NaiveBayes(
        featuresCol="features",
        labelCol="Accident_Severity_Binary"

    )

    nb_model = nb_classifier.fit(training_data)
    predictions = nb_model.transform(testing_data)

    classificationTest(predictions,testing_data,"Naive Bayes")


def gradientBoostedTrees(training_data, testing_data):

    gbtc_classifier = GBTClassifier(
        labelCol="Accident_Severity_Binary",
        featuresCol="features",
        maxBins=100,
        maxDepth= 12,
        seed=123)

    gbtc_model = gbtc_classifier.fit(training_data)
    predictions = gbtc_model.transform(testing_data)

    classificationTest(predictions,testing_data,"Gradient Boosted Trees")




def classificationTest(predictions, testing_data, model):

     print "\n"+model
     predictions.stat.crosstab("prediction", "Accident_Severity_Binary").show()
     matrix = predictions.stat.crosstab("prediction", "Accident_Severity_Binary").collect()

     TP = matrix[1]['0.0']
     FP = matrix[1]['1.0']
     TN = matrix[0]['1.0']
     FN = matrix[0]['0.0']

     # Metrics
     ACC = (TP+TN) / float(TP+TN+FP+FN)    # accuracy
     TPR = TP / float(TP+FN)   # recall
     PPV = TP / float(TP+FP)   # precision
     F1 = (2 * TP) / float(2*TP + FP + FN)
     evaluator = BinaryClassificationEvaluator(labelCol = 'Accident_Severity_Binary')

     # ROC
     AUC = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
     test_error = (FP + FN) / float(testing_data.count())

     print "ACC Accuracy: {0:.4f}".format(ACC)
     print "TPR Recall: {0:.4f}".format(TPR)
     print "TPV Precision: {0:.4f}".format(PPV)
     print "F1 Score: {0:.4f}".format(F1)
     print "Testing error: {0:.4f}".format(test_error)
     print "AUC: Area Under the ROC Curve: {0:.4f}\n".format(AUC)
