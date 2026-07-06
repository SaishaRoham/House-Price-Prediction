import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import joblib

data = pd.read_csv("dataset\house_prices.csv")

print(data.head())

print(data.info())

print(data.describe())

print(data.isnull().sum())
data = data.fillna(data.mean(numeric_only=True))

# Features (Independent Variables)
X = data[['Avg. Area Income',
          'Avg. Area House Age',
          'Avg. Area Number of Rooms',
          'Avg. Area Number of Bedrooms',
          'Area Population']]

# Target Variable (Dependent Variable)
y = data['Price']

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("Model trained successfully!")

# Predict house prices
predictions = model.predict(X_test)

print(predictions[:5])

score = r2_score(y_test, predictions)

print("R² Score:", score)
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", mae)

new_house = pd.DataFrame({
    "Avg. Area Income": [65000],
    "Avg. Area House Age": [6.5],
    "Avg. Area Number of Rooms": [7],
    "Avg. Area Number of Bedrooms": [4],
    "Area Population": [35000]
})

predicted_price = model.predict(new_house)

print("Predicted House Price: $", predicted_price[0])

print("Predicted House Price: $", predicted_price[0])

plt.figure(figsize=(8,6))

plt.scatter(y_test, predictions, alpha=0.7)

# Perfect prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red',
    linewidth=2,
    label='Perfect Prediction'
)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.legend()

plt.savefig("images/prediction_plot.png", dpi=300)

plt.show()

import joblib

joblib.dump(model, "model/house_price_model.pkl")

print("Model saved successfully!")

loaded_model = joblib.load("model/house_price_model.pkl")

prediction = loaded_model.predict(new_house)

print("Prediction from saved model:", prediction[0])

print("Prediction from saved model:", prediction[0])