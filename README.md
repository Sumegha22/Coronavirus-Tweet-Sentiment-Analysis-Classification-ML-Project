# Coronavirus-Tweet-Sentiment-Analysis-Classification-ML-Project



# 🎯 OBJECTIVE
Sentiment analysis on tweets related to the coronavirus (COVID-19) can be useful for understanding public opinion, tracking trends, and identifying potential areas of concern. In this task, we can use machine learning techniques, such as logistic regression or support vector machines, to predict the sentiment of a given tweet as positive, negative, or neutral.

# INTRODUCTION
Text classification with the help of machine learning can be achieved using many ML algorithm like
Logistic Regression with Grid Search CV

Decision Tree Classifier(Count Vector and TF ID Vectorization techniques).

KNN(Count Vector and TF ID Vectorization techniques).

SVM Classifier (Count Vector and TF ID Vectorization techniques).

Multinomial Navies Bayes & Bernoulli Navies Bayes(Count Vector and TF ID Vectorization techniques).

Random Forest(Count Vector and TF ID Vectorization techniques).

Stochastic Gradient Descent(Count Vector and TF ID Vectorization techniques).

# DATASET
Coronavirus Tweet Sentiment analysis dataset comprises of 41156 records which has 6 columns namely 'Location', 'Tweet At', 'Original Tweet', 'Sentiment','User name','Screen name'. In between 'Sentiment' column which tells about the inclination of the tweet in terms of 'positive', 'negetive', 'extremely positive','extremely negative'. So, here our target variable or dependent features is 'sentiment'.

# DATA PREPROCESSING
After loading the dataset and before building the models , the text should be cleaned and preprocessed to achieve better accuracy.
Data preprocessing means to change the data in a way that it is more effective while building a model by minimizing the least important features in the data.
It will include lowercasing, tokenization, punctuation removal, stopword removal like task.

# Feature Manipulation & Selection
It will include Extract relevant features from the data and transform them into a format that can be used for machine learning and Choose an appropriate machine learning algorithm or a combination of algorithms to solve the problem at hand. 

# Model training:
Split the data into training and testing sets, and train the model on the training set. Evaluate the model's performance on the testing set and tune the hyperparameters accordingly.Here, we use 80/20 ratio data spliting so get effective train & test data set.

# CONCLUSION
1.We applied 8 models namely, Logistic Regression with Grid Search CV, Decision Tree Classifier,Stochastic Gradient Descent , KNN, SVM,Multinomial Navies Bayes,Bernoulli Navies Bayes Classifier for both Count Vector And TF ID Vectorization techniques.

2..We conclude that the machine is generating the best results for the Stochastic Gradient Descent(count vectorizer) model with an Accuracy of 80.43% followed by the Logistic Regression with Grid Search CV (TF/ID vectorizer) model with an Accuracy of 78.86%.

3.In the future ,we can repeat the analysis and compare it with the present sentimental analysis to gauge the impact of the initiatives on the ground.




