import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# loading the data from csv file to a pandas Dataframe
raw_mail_data = pd.read_csv('mail_data.csv')

print(raw_mail_data)

# replace the null values with a null string
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')

# printing the first 5 rows of the dataframe
mail_data.head()

# checking the number of rows and columns in the dataframe
mail_data.shape

# label spam mail as 0;  ham mail as 1;
mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1

# separating the data as texts and label
X = mail_data['Message']
Y = mail_data['Category']

print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

print(X.shape)
print(X_train.shape)
print(X_test.shape)

# transform the text data to feature vectors that can be used as input to the Logistic regression
feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# convert Y_train and Y_test values as integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

print(X_train)
print(X_train_features)



model = LogisticRegression()
# training the Logistic Regression model with the training data
model.fit(X_train_features, Y_train)

# prediction on training data & testing data
prediction_on_training_data = model.predict(X_train_features)
prediction_on_test_data = model.predict(X_test_features)

# Acccuracy
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print('Accuracy on training data : ', accuracy_on_training_data)
print('Accuracy on test data : ', accuracy_on_test_data)


# Precision
precision_on_training_data = precision_score(Y_train, prediction_on_training_data)
precision_on_testing_data = precision_score(Y_test, prediction_on_test_data)
print("Precision on training data : ", precision_on_training_data)
print("Precision on testing data : ", precision_on_testing_data)


# Recall (Sensitivity)
Recall_on_training_data = recall_score(Y_train, prediction_on_training_data)
Recall_on_testing_data = recall_score(Y_test, prediction_on_test_data)
print("Recall on training data : ", Recall_on_training_data)
print("Recall on testing data : ", Recall_on_testing_data)


# F1-Score
F1Score_on_training_data = f1_score(Y_train, prediction_on_training_data)
F1Score_on_testing_data = f1_score(Y_test, prediction_on_test_data)
print("F1-Score on training data : ", F1Score_on_training_data)
print("F1-Score on testing data : ", F1Score_on_testing_data)


# Confusion Matrix
Confusion_on_training_data = confusion_matrix(Y_train, prediction_on_training_data)
Confusion_on_testing_data = confusion_matrix(Y_test, prediction_on_test_data)
print("Confusion on training data : ", Confusion_on_training_data)
print("Confusion on testing data : ", Confusion_on_testing_data)


input_mail = ["XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here>> http://wap. xxxmobilemovieclub.comn=QJKGIGHJJGCBL"]

# convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)

# making prediction
prediction = model.predict(input_data_features)
print(prediction)
if (prediction[0]==1):
  print('Ham mail')
else:
  print('Spam mail')
