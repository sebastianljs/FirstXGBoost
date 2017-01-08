# First XGBoost model for Pima Indian dataset

# Load relevant packages

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

pima = loadtxt('pima-indians-diabetes.csv', delimiter=',')

# Split data into features and target

features = pima[:,0:8]
target = pima[:,8]

# Split data into train and test sets

seed = 7
test_size = 0.33
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=test_size, random_state=seed)

# Fit model with training data

model = XGBClassifier()
model.fit(features_train, target_train)

# Make predictions on testing data

target_pred = model.predict(features_test)

# Round off predictions (Since the predictions are binary)

predictions = [round(value) for value in target_pred]

accuracy = accuracy_score(target_test, predictions)

# Output with accuracy up to 2 decimal places in percentage
print("{:.2%}".format(accuracy))

# AWS API Interface Test

import boto3

s3 = boto3.resource('s3')
for bucket in s3.buckets.all():
    print(bucket.name)