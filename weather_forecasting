# ---------------- DATASET ----------------
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------- DATASET ----------------
data = {
    'Day': [1,2,3,4,5,6,7,8,9,10],
    'Temp': [30,32,31,29,28,30,31,33,34,32],
    'Humidity': [70,65,80,75,85,68,72,60,58,66],
    'Wind': [10,12,8,9,7,11,10,14,13,12],
    'Rain': [1,0,1,1,1,0,1,0,0,0]
}

df = pd.DataFrame(data)
print("Dataset:\n", df)

# ---------------- TEMPERATURE PREDICTION ----------------
X_temp = df[['Day']]
y_temp = df['Temp']

temp_model = LinearRegression()
temp_model.fit(X_temp, y_temp)

day = int(input("Enter future day to predict temperature: "))
temp_prediction = temp_model.predict([[day]])

print(f"Predicted Temperature for day {day}: {temp_prediction[0]:.2f}")

# ---------------- TEMPERATURE GRAPH ----------------
plt.scatter(X_temp, y_temp, label="Actual Temp")
plt.plot(X_temp, temp_model.predict(X_temp), label="Regression Line")
plt.scatter(day, temp_prediction[0], color='red', label="Predicted Temp")

plt.xlabel("Day")
plt.ylabel("Temperature")
plt.title("Temperature Prediction")
plt.legend()
plt.show()

# ---------------- RAIN PREDICTION ----------------
X_rain = df[['Temp','Humidity','Wind']]
y_rain = df['Rain']

X_train, X_test, y_train, y_test = train_test_split(X_rain, y_rain, test_size=0.2)

rain_model = RandomForestClassifier()
rain_model.fit(X_train, y_train)

# Accuracy
y_pred = rain_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# User input for rain prediction
humidity = float(input("Enter humidity: "))
wind = float(input("Enter wind speed: "))

rain_prediction = rain_model.predict([[temp_prediction[0], humidity, wind]])

if rain_prediction[0] == 1:
    print("Rain Expected 🌧️")
else:
    print("No Rain ☀️")

