import numpy as np
import pandas as pd
from Bag_of_words import BOW, y, BOW_test
from Bernoulli import bernoulli, bernoulli_test
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from imutils import paths
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from MCAP_logistic_regression import bow_data_loading, bernoulli_data_loading

# Importing the y values of training data.
Y = y()

# Defining the list of path where all the test data files are stored.
filePaths = (list(paths.list_files("enron1/test/ham")) + list(paths.list_files("enron4/test/ham")) + list(paths.list_files("hw1/test/ham"))
             + list(paths.list_files("enron1/test/spam")) + list(paths.list_files("enron4/test/spam")) + list(paths.list_files("hw1/test/spam")))


if __name__ == "__main__":
    # Defining the sklearn SGDClassifier
    sgd = SGDClassifier()

    # Declaring the dictionary containg the parameters for the SGDClassifier.
    parameters = {"loss": ["log"], "penalty": ["l2"],
                  "max_iter": [100], "learning_rate": ['constant'], "eta0": [0.002]}

    # Using the GridSearchCV to fit the SGDClassifier on the training data and to tune the parameters.
    model = GridSearchCV(sgd, param_grid=parameters)

    # Importing the bag of words training data from MCAP_logistic_regression.py file
    _, _, _, _, bow_train = bow_data_loading()

    # Importing the bag of words testing data from Bag_of_words.py file.
    bow_test, bow_test_y = BOW_test(filePaths)

    # Training the model on bag of words data, predicting the label on test data and checking the accuracy.
    model.fit(bow_train, Y)

    #print("Best parameters for bag of words: {}".format(model.best_params_))

    # Training the model on bag of words data, predicting the label on test data and checking the accuracy.
    bow_predicted_y = model.predict(bow_test)
    print("Accuracy on the bag of words data: {}".format(
        accuracy_score(bow_predicted_y, bow_test_y)))

    PRFS = precision_recall_fscore_support(
        bow_test_y, bow_predicted_y, average="macro")
    print("Precision for bag_of_words is {}".format(PRFS[0]))
    print("Recall for bag_of_words is {}".format(PRFS[1]))
    print("F1 score for bag_of_words is {}".format(PRFS[2]))

    # Importing the bag of words training data from MCAP_logistic_regression.py file
    _, _, _, _, bernoulli_train = bernoulli_data_loading()

    # Importing the bag of words testing data from benoulli.py file.
    bernoulli_test, bernoulli_test_y = bernoulli_test(filePaths)

    # Training the model on bag of words data, predicting the label on test data and checking the accuracy.
    model.fit(bernoulli_train, Y)
    #print("Best parameters for bernoulli: {}".format(model.best_params_))

    bernoulli_predicted_y = model.predict(bernoulli_test)
    print("Accuracy on the bernoulli data: {}".format(
        accuracy_score(bernoulli_predicted_y, bernoulli_test_y)))

    PRFS = precision_recall_fscore_support(
        bernoulli_test_y, bernoulli_predicted_y, average="macro")
    print("Precision for bernoulli is {}".format(PRFS[0]))
    print("Recall for bernoulli is {}".format(PRFS[1]))
    print("F1 score for bernoulli is {}".format(PRFS[2]))
