import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------- DATASET ----------------
data = {
    'Age': [25, 45, 35, 50, 23, 40, 60, 55, 30, 48],
    'BP': [120, 140, 130, 150, 110, 135, 160, 155, 125, 145],
    'Cholesterol': [200, 240, 220, 260, 180, 230, 270, 250, 210, 245],
    'Disease': [0, 1, 0, 1, 0, 1, 1, 1, 0, 1]   # 1 = Yes, 0 = No
}

df = pd.DataFrame(data)
print("Dataset:\n", df)

# ---------------- FEATURES & TARGET ----------------
X = df[['Age','BP','Cholesterol']]
y = df['Disease']

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# ---------------- MODEL ----------------
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)

# ---------------- PREDICTION ----------------
y_pred = model.predict(X_test)

# ---------------- ACCURACY ----------------
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# ---------------- USER INPUT ----------------
age = int(input("Enter Age: "))
bp = int(input("Enter Blood Pressure: "))
chol = int(input("Enter Cholesterol: "))

prediction = model.predict([[age, bp, chol]])

if prediction[0] == 1:
    print("Disease Detected ⚠️")
else:
    print("No Disease ✅")

# ---------------- TREE GRAPH ----------------
plt.figure(figsize=(12,6))
plot_tree(
    model,
    feature_names=['Age','BP','Cholesterol'],
    class_names=['No Disease','Disease'],
    filled=True
)
plt.title("Decision Tree Visualization")
plt.show()
