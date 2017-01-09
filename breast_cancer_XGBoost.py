
# coding: utf-8

# binary classification, breast cancer dataset, label and one hot encoded
from numpy import column_stack
from pandas import read_csv
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# load data and drop missing values

data = read_csv('breast-cancer.csv', header=None)
data = data.dropna()
data = data.values


# split data into X and y

features = data[:, 0:9]
target = data[:, 9]


# Encode string input values as integers

columns = []
for i in range(0, features.shape[1]):
    label_encoder = LabelEncoder()
    feature = label_encoder.fit_transform(features[:, i])
    # Reshape feature into one column
    feature = feature.reshape(features.shape[0], 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    feature = onehot_encoder.fit_transform(feature)
    columns.append(feature)

features_encoded = column_stack(columns)
print("Features shape: ", features_encoded.shape)


# Encode string class values as integers

label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)


# Split data into training and testing datasets

seed = 7
test_size = 0.33
features_train, features_test, target_train, target_test =     train_test_split(features_encoded, target_encoded, test_size=test_size, random_state=seed)


# Fit model using training data

model = XGBClassifier()
model.fit(features_train, target_train)
print(model)


# Make predictions using testing data

target_pred = model.predict(features_test)
predictions = [round(value) for value in target_pred]


# Evaluate predictions

accuracy = accuracy_score(target_test, predictions)
print("{:.2%}".format(accuracy))

