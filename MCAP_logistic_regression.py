import numpy as np
from sklearn.model_selection import train_test_split
from Bag_of_words import BOW, y, BOW_test
from Bernoulli import bernoulli, bernoulli_test
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from imutils import paths

# Importing the y values of training data.
Y = y()

# Defining the learning rate and number of iterations for the trainig purpose.
learning_rate = 0.001
epochs = 200

# Defining the list of path where all the test data files are stored.
filePaths = (list(paths.list_files("Datasets/enron1/test/ham")) + list(paths.list_files("Datasets/enron4/test/ham")) + list(paths.list_files("Datasets/hw1/test/ham"))
             + list(paths.list_files("Datasets/enron1/test/spam")) + list(paths.list_files("Datasets/enron4/test/spam")) + list(paths.list_files("Datasets/hw1/test/spam")))


# Function to load the bag of words data from the Bag_of_words.py file and for splitting the data in the 70/30.
def bow_data_loading():
    bag_of_words, _ = BOW()
    bow_x_train, bow_x_test, bow_y_train, bow_y_test = train_test_split(
        bag_of_words, Y, test_size=0.3, shuffle=True)

    return bow_x_train, bow_x_test, bow_y_train, bow_y_test, bag_of_words


# Function to load the bernoulli data from the Bernoulli.py file and for splitting the data in the 70/30.
def bernoulli_data_loading():
    ham_data, spam_data = bernoulli()
    bernoulli_data = np.concatenate((ham_data, spam_data), axis=0)
    #bernoulli_y = y()
    # print(bernoulli_y)
    ber_x_train, ber_x_test, ber_y_train, ber_y_test = train_test_split(
        bernoulli_data, Y, test_size=0.3, shuffle=True)

    return ber_x_train, ber_x_test, ber_y_train, ber_y_test, bernoulli_data


# Function to predict the class label using sigmoid.
def predict(w, x, w0):
    z = np.sum(np.multiply(w, x), axis=1) + w0
    sigmoid = 1/(1+np.exp(-z))
    return sigmoid


# Function to train the bag of words data using logistic regression.
def bow_logistic_regresion():
    # Lambda value
    lam = 0.1

    bow_x_train, bow_x_test, bow_y_train, bow_y_test, bag_of_words_train = bow_data_loading()
    w = np.zeros(len(bow_x_train[0]))
    w0 = 0
    for epoch in range(epochs):
        # Predicted value of y
        y_pred = predict(w, bag_of_words_train, w0)

        # Calculating the derivative of loss function ans updating the weights.
        derivative_0 = np.subtract(Y, y_pred)
        derivatives = np.dot(np.transpose(bag_of_words_train), derivative_0)
        derivative_0 = np.sum(derivative_0)

        w0 = w0 + learning_rate * (derivative_0 - lam*w0)
        w = w + learning_rate * (derivatives - lam*w)

    return w0, w


# Function to train the bernoulli data using logistic regression.
def bernoulli_logistic_regresion():
    # Lambda value
    lam = 0.5

    bernoulli_x_train, bernoulli_x_test, bernoulli_y_train, bernoulli_y_test, bernoulli_train = bernoulli_data_loading()
    w = np.zeros(len(bernoulli_x_train[0]))
    w0 = 0
    for epoch in range(epochs):
        # Predicted value of y
        y_pred = predict(w, bernoulli_train, w0)

        # Calculating the derivative of loss function ans updating the weights.
        derivative_0 = np.subtract(Y, y_pred)
        derivatives = np.dot(np.transpose(bernoulli_train), derivative_0)
        derivative_0 = np.sum(derivative_0)

        w0 = w0 + learning_rate * (derivative_0 - lam*w0)
        w = w + learning_rate * (derivatives - lam*w)

    return w0, w


if __name__ == "__main__":

    # Checking the accuracy of the model on bag of words test dataset.
    #bow_x_train, bow_x_test, bow_y_train, bow_y_test, _ = bow_data_loading()
    bow_test, bow_test_y = BOW_test(filePaths)
    bow_w0, bow_w = bow_logistic_regresion()
    bow_Y = predict(bow_w, bow_test, bow_w0)
    bow_y_pred = []
    for y in bow_Y:
        if y >= 0.5:
            bow_y_pred.append(1)
        else:
            bow_y_pred.append(0)

    bow_accuracy = accuracy_score(bow_y_pred, bow_test_y)
    print("Accuracy for bag of words is {}".format(bow_accuracy))

    PRFS = precision_recall_fscore_support(
        bow_test_y, bow_y_pred, average="macro")
    print("Precision for bag of words is {}".format(PRFS[0]))
    print("Recall for bag of words is {}".format(PRFS[1]))
    print("F1 score for bag of words is {}".format(PRFS[2]))

    # Checking the accuracy of the model on bernoulli test dataset.
    #bernoulli_x_train, bernoulli_x_test, bernoulli_y_train, bernoulli_y_test, _ = bernoulli_data_loading()
    bernoulli_test, bernoulli_test_y = bernoulli_test(filePaths)

    bernoulli_w0, bernoulli_w = bernoulli_logistic_regresion()
    bernoulli_Y = predict(bernoulli_w, bernoulli_test, bernoulli_w0)

    bernoulli_y_pred = []
    for y in bernoulli_Y:
        if y >= 0.5:
            bernoulli_y_pred.append(1)
        else:
            bernoulli_y_pred.append(0)

    bernoulli_accuracy = accuracy_score(bernoulli_y_pred, bernoulli_test_y)
    print("Accuracy for bernoulli is {}".format(bernoulli_accuracy))

    PRFS = precision_recall_fscore_support(
        bernoulli_test_y, bernoulli_y_pred, average="macro")
    print("Precision for bernoulli is {}".format(PRFS[0]))
    print("Recall for bernoulli is {}".format(PRFS[1]))
    print("F1 score for bernoulli is {}".format(PRFS[2]))
