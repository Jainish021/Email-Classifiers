from Bag_of_words import y, ham, spam, BOW_test
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from imutils import paths


# Function to calculate prior probabilities of ham and spam.
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


# Function to calculate probabilities of words in ham and spam dataset, and also applying one laplace smoothing.
def word_probs():
    ham_data, _ = ham()
    ham_sum = ham_data.sum(axis=0) + 1
    ham_total = ham_sum.sum(axis=0) + 2
    ham_word_prob = np.divide(ham_sum, ham_total)
    ham_word_prob = np.log(ham_word_prob)

    spam_data, _ = spam()
    spam_sum = spam_data.sum(axis=0) + 1
    spam_total = spam_sum.sum(axis=0) + 2
    spam_word_prob = np.divide(spam_sum, spam_total)
    spam_word_prob = np.log(spam_word_prob)

    return ham_word_prob, spam_word_prob


def predict():
    # Defining the list of path where all the test data files are stored.
    filePaths = (list(paths.list_files("Datasets/enron1/test/ham")) + list(paths.list_files("Datasets/enron4/test/ham")) + list(paths.list_files("Datasets/hw1/test/ham"))
                 + list(paths.list_files("Datasets/enron1/test/spam")) + list(paths.list_files("Datasets/enron4/test/spam")) + list(paths.list_files("Datasets/hw1/test/spam")))

    # Creating a test dataset using BOW_test method of Bag_of_words.py file.
    test_data, true_y = BOW_test(filePaths)

    y = []
    # Getting the values from other functions.
    ham_word_prob, spam_word_prob = word_probs()
    prior_ham, prior_spam = prior()

    # Proccesing the test data predicting the class label.
    for i in test_data:
        log_prob_ham = np.sum(np.multiply(ham_word_prob, i))
        log_prob_spam = np.sum(np.multiply(spam_word_prob, i))

        ham_prediction = prior_ham * np.exp(log_prob_ham)
        spam_prediction = prior_spam * np.exp(log_prob_spam)

        if ham_prediction > spam_prediction:
            y.append(0)
        else:
            y.append(1)
    return y, true_y


if __name__ == "__main__":
    p_h, p_s = prior()
    word_probs()
    y_predicted, true_y = predict()
    result = accuracy_score(y_predicted, true_y)
    print("Accuracy is {}".format(result))

    PRFS = precision_recall_fscore_support(
        true_y, y_predicted, average="macro")
    print("Precision is {}".format(PRFS[0]))
    print("Recall is {}".format(PRFS[1]))
    print("F1 score is {}".format(PRFS[2]))
