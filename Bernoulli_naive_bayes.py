#Code has some logical error.


from os import name
from Bag_of_words import y, ham, spam
from Bernoulli import bernoulli, bernoulli_test
import numpy as np
from imutils import paths
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def prior():
    Y = y()
    prior_ham = prior_spam = 0
    for i in Y:
        if i == 0:
            prior_ham += 1
        else:
            prior_spam += 1

    prior_ham = prior_ham/len(Y)
    prior_spam = prior_spam/len(Y)

    return prior_ham, prior_spam


def word_probs():
    ham_data, spam_data = bernoulli()

    ham_sum = ham_data.sum(axis=0) + 1
    ham_total = len(ham_data) + np.sum(ham_sum)
    ham_word_prob = np.divide(ham_sum, ham_total)
    #ham_word_prob = np.log(ham_word_prob)

    spam_sum = spam_data.sum(axis=0) + 1
    spam_total = len(spam_data) + np.sum(spam_sum)
    spam_word_prob = np.divide(spam_sum, spam_total)
    #spam_word_prob = np.log(spam_word_prob)

    return ham_word_prob, spam_word_prob


def numerator_values():
    prior_ham, prior_spam = prior()
    ham_word_prob, spam_word_prob = word_probs()
    ham_numerator = np.sum(ham_word_prob) + np.log(prior_ham)
    # print(ham_numerator)
    spam_numerator = np.sum(spam_word_prob) + np.log(prior_spam)
    # print(spam_numerator)
    return ham_numerator, spam_numerator


def predict():
    # Defining the list of path where all the test data files are stored.
    filePaths = (list(paths.list_files("Datasets/enron1/test/ham")) + list(paths.list_files("Datasets/enron4/test/ham")) + list(paths.list_files("Datasets/hw1/test/ham"))
                 + list(paths.list_files("Datasets/enron1/test/spam")) + list(paths.list_files("Datasets/enron4/test/spam")) + list(paths.list_files("Datasets/hw1/test/spam")))

    # Creating a test dataset using BOW_test method of Bag_of_words.py file.
    test_data, true_y = bernoulli_test(filePaths)

    y = []
    ham_word_prob, spam_word_prob = word_probs()
    ham_numerator, spam_numerator = numerator_values()

    for i in test_data:
        temp1 = np.multiply(i, ham_word_prob) + \
            np.multiply((1 - i), (1 - spam_word_prob))
        temp1 = np.sum(np.log(temp1))
        temp2 = np.multiply(i, spam_word_prob) + \
            np.multiply((1 - i), spam_word_prob)
        temp2 = np.sum(np.log(temp2))

        if np.exp(temp1) >= np.exp(temp2):
            y.append(0)
        else:
            y.append(1)
    return y, true_y


if __name__ == "__main__":
    prior()
    word_probs()
    numerator_values()
    y_predicted, true_y = predict()
    print("Accuracy is {}".format(accuracy_score(y_predicted, true_y)))

    PRFS = precision_recall_fscore_support(
        true_y, y_predicted, average="macro")
    print("Precision is {}".format(PRFS[0]))
    print("Recall is {}".format(PRFS[1]))
    print("F1 score is {}".format(PRFS[2]))
