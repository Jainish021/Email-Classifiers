# Email-Classifiers
Python version used - 3.9.7

All the files are well commented to explain the working of code. Following is the brief description about what that particular file contains. Simply run the file to get the accuracy, precision, recall and f1 score values.

=> Vocabulary.py
	This file is for preprocessing of the data and to create the vocabulary of words. Run this file to see the vocabulary used.

=> Bag_of_words.py
	This file is used to convert the training and testing data of all the three datasets into one matrix of features*example using bag of words model.

=> Bernoulli.py
	This file is used to convert the training and testing data of all the three datasets into one matrix of features*example using bernoulli model.

=> Multinomial_naive_bayes.py
	This file has the code for multinomial naive bayes. Run this file to see the accuracy, precision, recall and f1 score values of the trained model on the test data.
	
	Results:
	Accuracy is 0.8781
	Precision is 0.8843
	Recall is 0.8851
	F1 score is 0.8781

=> Bernoulli_naive_bayes.py
	This file has the code for discrete (bernoulli) naive bayes. Run this file to see the accuracy, precision, recall and f1 score values of the trained model on the test data.

=> MCAP_logistic_regression.py
	This file has the code for MCAP Logistic Regression algorithm with L2 regularization. Run this file to see the accuracy, precision, recall and f1 score values of the trained model on the test data.
	
	Results:
	Accuracy - 0.9627
	Precision - 0.9615
	Recall - 0.9644
	F1 score - 0.9625
