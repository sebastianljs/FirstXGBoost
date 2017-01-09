
# coding: utf-8

# Load relevant packages 

from pandas import read_csv
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


# Load dataset

iris = read_csv('iris.csv', header = None).values


# Split dataset into features and targets 

features = iris[:,0:-1]
target = iris[:,-1]


# Encode iris class strings as integers

label_encoder = LabelEncoder()
label_encoded_target = label_encoder.fit_transform(target)


# Split data in training and testing sets

seed = 7
test_size = 0.33
features_train, features_test, target_train, target_test     = model_selection.train_test_split(features, label_encoded_target, test_size = test_size, random_state=seed)


# Fit model into training data

model = XGBClassifier()
model.fit(features_train, target_train)
print(model)


# Make predictions for testing data

target_pred = model.predict(features_test)
predictions = [round(value) for value in target_pred]


# Evaluate predictions

accuracy = accuracy_score(target_test, predictions)
print("{:.2%}".format(accuracy))

