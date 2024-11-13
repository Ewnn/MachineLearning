import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, RFECV
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

# Load Titanic data (replace with your actual Titanic CSV file path)
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# Display the first few rows of the dataset
print(test_data.head())

# Show the shape of the dataset
print(f'Dataset Shape: {test_data.shape}')

# Plot missing values using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(test_data.isnull(), cbar=False, cmap='viridis')
plt.show()

# Print the percentage of missing values in the 'Age' column
print(f'Age missing values: {test_data["Age"].isnull().mean() * 100:.2f}%')

# Plot histogram of 'Age' column
plt.figure(figsize=(8, 6))
sns.histplot(test_data['Age'].dropna(), kde=True)
plt.title('Age Distribution')
plt.show()

# Print the mean and median of 'Age' column
print(f'Mean Age: {test_data["Age"].mean():.2f}')
print(f'Median Age: {test_data["Age"].median():.2f}')

# Print the percentage of missing values in the 'Cabin' column
print(f'Cabin missing values: {test_data["Cabin"].isnull().mean() * 100:.2f}%')

# Print the distribution of 'Embarked' column
print(test_data['Embarked'].value_counts())

# Impute missing values in 'Age' with the median and 'Embarked' with the most frequent value
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
test_data['Embarked'].fillna(test_data['Embarked'].mode()[0], inplace=True)

# Drop the 'Cabin' column due to too many missing values
test_data.drop(columns=['Cabin'], inplace=True)

# Verify the null values
print(test_data.isnull().sum())

# Feature Engineering: Create the 'TravelAlone' feature
test_data['TravelAlone'] = (test_data['SibSp'] + test_data['Parch'] == 0).astype(int)

# Use get_dummies to encode categorical features
test_data_encoded = pd.get_dummies(test_data, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)

# Drop the 'Name' and 'Ticket' columns (don't need them for prediction)
test_data_encoded.drop(columns=['Name', 'Ticket'], inplace=True)

# Display the processed dataframe
print(test_data_encoded.head())

# Define X (features) and y (target)
X = test_data_encoded.drop(columns=['Survived'])
y = test_data_encoded['Survived']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate and train a Logistic Regression model
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lr.predict(X_test)

# Evaluate the model with accuracy, log_loss, and AUC
accuracy = accuracy_score(y_test, y_pred)
logloss = log_loss(y_test, lr.predict_proba(X_test))
auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])

print(f'Accuracy: {accuracy:.2f}')
print(f'Log Loss: {logloss:.2f}')
print(f'AUC: {auc:.2f}')

# Feature Selection with Recursive Feature Elimination (RFE)
rfe = RFE(lr, n_features_to_select=4)
rfe.fit(X_train, y_train)

# Print selected features
print(f'Selected features (4): {X.columns[rfe.support_]}')

# Feature Selection with RFECV (RFE with cross-validation)
rfecv = RFECV(estimator=lr, step=1, cv=5, scoring='accuracy')
rfecv.fit(X_train, y_train)

# Print the optimal number of features and the selected features
print(f'Optimal number of features: {rfecv.n_features_}')
print(f'Selected features (RFECV): {X.columns[rfecv.support_]}')

# Plot the number of features vs. cross-validation scores
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.xlabel('Number of features')
plt.ylabel('Cross-validation score')
plt.title('Feature Selection with RFECV')
plt.show()

# Re-train the model using the selected features from RFECV
X_train_selected = X_train[X.columns[rfecv.support_]]
X_test_selected = X_test[X.columns[rfecv.support_]]

lr.fit(X_train_selected, y_train)

# Make predictions again and evaluate the model
y_pred_selected = lr.predict(X_test_selected)
accuracy_selected = accuracy_score(y_test, y_pred_selected)
logloss_selected = log_loss(y_test, lr.predict_proba(X_test_selected))
auc_selected = roc_auc_score(y_test, lr.predict_proba(X_test_selected)[:, 1])

print(f'Accuracy with selected features: {accuracy_selected:.2f}')
print(f'Log Loss with selected features: {logloss_selected:.2f}')
print(f'AUC with selected features: {auc_selected:.2f}')
