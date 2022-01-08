from Vocabulary import vocab, extract
import pandas as pd
from imutils import paths
from sklearn.feature_extraction.text import CountVectorizer


# Importing the vocabulary built in Vocabulary.py file
vocabulary = vocab()

# Defining the list of path where all the training data files are stored.
filePaths = (list(paths.list_files("enron1/train/ham")) + list(paths.list_files("enron4/train/ham")) + list(paths.list_files("hw1/train/ham"))
             + list(paths.list_files("enron1/train/spam")) + list(paths.list_files("enron4/train/spam")) + list(paths.list_files("hw1/train/spam")))

# Reading all training data file and preparing a corresponding features*example matrix for bag_of_words model using CountVectorizer() method.
vec = CountVectorizer()


def BOW():
    # Initializing the data list
    data = []
    # Reading all training data file and preparing a corresponding features*example matrix for bag_of_words model using CountVectorizer() method.
    vec = CountVectorizer()
    for filePath in filePaths:
        with open(filePath, errors="ignore") as f:
            contents = f.read()
            data.append(contents)
            f.close()
    vec.fit(vocabulary)
    mat = vec.transform(data)
    # Creatin the dataframe of the data matrix.
    cols = vec.get_feature_names()
    rows = range(len(mat.toarray()))
    df = pd.DataFrame(mat.toarray(), columns=cols, index=rows)
    return mat.toarray(), df


def BOW_test(filePaths):
    # Initializing the data list
    data = []
    true_y = []

    for filePath in filePaths:
        with open(filePath, errors="ignore") as f:
            contents = f.read()
            data.append(contents)
            if "ham" in filePath:
                true_y.append(0)
            else:
                true_y.append(1)
            f.close()
    vec.fit(vocabulary)
    mat = vec.transform(data)
    return mat.toarray(), true_y


# Defining a ham function to create a features*example matrix of ham training data.
def ham():
    # Defining the list of path where all the ham training data files are stored.
    filePaths = (list(paths.list_files("enron1/train/ham")) + list(
        paths.list_files("enron4/train/ham")) + list(paths.list_files("hw1/train/ham")))

    data = []

    # Reading all training data file and preparing a corresponding features*example matrix for bag_of_words model using CountVectorizer() method.
    for filePath in filePaths:
        with open(filePath, errors="ignore") as f:
            contents = f.read()
            data.append(contents)
            f.close()
    vec.fit(vocabulary)
    mat = vec.transform(data)

    cols = vec.get_feature_names()
    rows = range(len(mat.toarray()))
    df = pd.DataFrame(mat.toarray(), columns=cols, index=rows)
    return mat.toarray(), df


# Defining a spam function to create a features*example matrix of spam training data.
def spam():
    # Defining the list of path where all the spam training data files are stored.
    filePaths = (list(paths.list_files("enron1/train/spam")) + list(
        paths.list_files("enron4/train/spam")) + list(paths.list_files("hw1/train/spam")))

    data = []

    # Reading all training data file and preparing a corresponding features*example matrix for bag_of_words model using CountVectorizer() method.
    for filePath in filePaths:
        with open(filePath, errors="ignore") as f:
            contents = f.read()
            data.append(contents)
            f.close()
    vec.fit(vocabulary)
    mat = vec.transform(data)

    cols = vec.get_feature_names()
    rows = range(len(mat.toarray()))
    df = pd.DataFrame(mat.toarray(), columns=cols, index=rows)
    return mat.toarray(), df


# Creating a vector of given classes in the training dataset.
def y():
    class_data = []
    for filePath in filePaths:
        if "ham" in filePath:
            class_data.append(0)
        else:
            class_data.append(1)
    return class_data


if __name__ == "__main__":
    y = y()
    h, df_h = ham()
    print(h)
    s, df_s = spam()
    print(s)
    bow = BOW()
