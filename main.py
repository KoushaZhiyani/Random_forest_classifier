import pandas as pd
from sklearn.model_selection import train_test_split
from random_forest import Random_Forest


data = pd.read_csv("PCOS.csv")

data = data.drop(data.index[200:], axis=0).dropna()  # Remove NaN values
X = data.drop(["PCOS (Y/N)"], axis=1)  # Features
y = data[["PCOS (Y/N)"]]  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=12)  # Train-test split

model = Random_Forest()
model.fit_model(X_train, y_train)
model.make_predict(X_test)
print(y_pre)
accuracy = model.score(y_pre, y_test)
print("Accuracy:", accuracy)
