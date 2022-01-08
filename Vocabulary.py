import numpy as np
from imutils import paths
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

porter = PorterStemmer()


# Defining a function to remove stopwords, numbers, special characters etc. from our data in order extract only words for our vocabulary.
# Also applying stemming to the database words
def extract(contents):
    ignore = set(stopwords.words('english'))
    words = re.sub("[^\w]|[\_]|[\d]", " ",  contents).split()
    data = []
    for word in words:
        if word not in ignore:
            data.append(porter.stem(word))
    return data


# Importing the data from all the training file, preproccessing the data using extract function and creating a vocalubulary.
def vocab():
    filePaths = (list(paths.list_files("enron1/train/ham")) +
                 list(paths.list_files("enron1/train/spam")) + list(paths.list_files("enron4/train/ham")) + list(paths.list_files("enron4/train/spam")) + list(paths.list_files("hw1/train/ham")) + list(paths.list_files("hw1/train/spam")))

    vocabulary = set()
    for filePath in filePaths:
        with open(filePath, errors="ignore") as f:
            contents = f.read()
            vocabulary.update(extract(contents))
    return sorted(vocabulary)


if __name__ == "__main__":
    vocabulary = vocab()
    print(vocabulary)
