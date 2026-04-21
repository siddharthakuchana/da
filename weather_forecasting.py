import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
df = fetch_openml(name='weather', version=1, as_frame=True).frame

# Drop missing values
df = df.dropna()

# Convert categorical columns to numeric using One-Hot Encoding
df = pd.get_dummies(df)

# Split features and target
X, y = df.iloc[:, :-1], df.iloc[:, -1]

# Train-test split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(Xtr, ytr)

# Prediction
pred = model.predict(Xte)

# RMSE
print("RMSE:", np.sqrt(mean_squared_error(yte, pred)))

# Result table
res = pd.DataFrame({
    'Actual': yte.values[:10],
    'Predicted': pred[:10]
})
res['Error'] = res['Actual'] - res['Predicted']
print("\nWeather Forecast Table:\n", res.round(2))

# Plot
plt.plot(yte.values[:50], label='Actual')
plt.plot(pred[:50], label='Predicted')
plt.legend()
plt.title("Weather Forecast Curve")
plt.xlabel("Samples")
plt.ylabel("Value")
plt.show()