from Vocabulary import vocab
from Bag_of_words import ham, spam
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


# Importing the vocabulary built in Vocabulary.py file
vocabulary = vocab()

# Reading all training data file and preparing a corresponding features*example matrix for bag_of_words model using CountVectorizer() method.
vec = CountVectorizer()


# Converting the dataframe created in the bag of words module to the matrix of bernoulli model using masking method.
def bernoulli():
    _, ham_data = ham()
    _, spam_data = spam()

    ham_data = ham_data.mask(ham_data > 0, 1)
    spam_data = spam_data.mask(ham_data > 0, 1)
    # print(ham_data)
    return ham_data.to_numpy(), spam_data.to_numpy()


# Preproccesing the test data and converting it into bernoulli model data.
def bernoulli_test(filePaths):
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

    cols = vec.get_feature_names()
    rows = range(len(mat.toarray()))
    df = pd.DataFrame(mat.toarray(), columns=cols, index=rows)

    test_data = df.mask(df > 0, 1)
    return test_data.to_numpy(), true_y


if __name__ == "__main__":
    ber_h, ber_s = bernoulli()
    print(ber_h)
    print(ber_s)
