# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from random_forest import Random_Forest  # Import custom Random Forest class
import random

# Set random seed for reproducibility
random.seed(10)

# Read the dataset
data = pd.read_csv("PCOS.csv")

# Split the dataset into features (X) and target variable (y)
X = data.drop(["PCOS (Y/N)"], axis=1)  # Features
y = data[["PCOS (Y/N)"]]  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the Random Forest model with specified parameters
model = Random_Forest(n_estimators=3, max_depth=5)

# Train the model using the training sets
model.fit_model(X_train, y_train)

# Make predictions on the test set
y_pred = model.make_predict(X_test)

# Evaluate the model accuracy
accuracy = model.score(y_pred, y_test)

# Print the accuracy
print("Accuracy:", accuracy)
