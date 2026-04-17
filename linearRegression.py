import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# ---------------- DATASET ----------------
data = {
    'Day': [1,2,3,4,5,6,7],
    'Temp': [30,32,31,29,28,30,31]
}

df = pd.DataFrame(data)
print("Dataset:\n", df)

# ---------------- FEATURES & TARGET ----------------
X = df[['Day']]
y = df['Temp']

# ---------------- MODEL ----------------
model = LinearRegression()
model.fit(X, y)

# ---------------- PREDICTION ----------------
day = int(input("Enter day to predict temperature: "))
prediction = model.predict([[day]])

print(f"Predicted Temperature for day {day}: {prediction[0]:.2f}")

# ---------------- ERROR ----------------
y_pred = model.predict(X)
mae = mean_absolute_error(y, y_pred)
print("Mean Absolute Error:", mae)

# ---------------- GRAPH ----------------
plt.scatter(X, y, label="Actual Data")
plt.plot(X, y_pred, label="Regression Line")
plt.scatter(day, prediction[0], color='red', label="Predicted Point")

plt.xlabel("Day")
plt.ylabel("Temperature")
plt.title("Linear Regression - Temperature Prediction")
plt.legend()
plt.show()
