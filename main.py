import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, log_loss

# Read the data
data = pd.read_csv('iris.csv')

# Separate features and target
X = data.drop('variety', axis=1)
y = data['variety']

# Split the data - 70% training, 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the QDA model
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train_scaled, y_train)

# Make predictions
y_pred = qda.predict(X_test_scaled)
y_pred_proba = qda.predict_proba(X_test_scaled)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
loss = log_loss(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"Log Loss: {loss:.4f}")