import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
data = {
    'Age': [25, 45, 35, 50, 23, 40, 60, 55, 30, 48],
    'BP': [120, 140, 130, 150, 110, 135, 160, 155, 125, 145],
    'Cholesterol': [200, 240, 220, 260, 180, 230, 270, 250, 210, 245],
    'Disease': [0, 1, 0, 1, 0, 1, 1, 1, 0, 1]   # 1 = Yes, 0 = No
}
df=pd.DataFrame(data)
print("dataset:\n",df)

X=df[['Age','BP','Cholesterol']]
y=df['Disease']
X_train,X_test,y_train,y_test=train_test_split(X_train, y_train, test_size=0.2, random_state=42)
model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy: ",accuracy)
cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:\n",cm)
age=int(input("enter the age: "))
bp=int(input("enter the bp: "))
cholesterol=int(input("enter the cholesterol: "))
prediction=model.predict([[age,bp,cholesterol]])
if prediction[0]==1:
    print("Disease: Yes")
else:
    print("Disease: No")

plt.scatter(range(len(y_test)),y_test,label="actual")
plt.scatter(range(len(y_pred)),y_pred,label="predicted")
plt.xlabel("TEST DATA INDEX")
plt.ylabel("Class(0/1)")
plt.title("Logistic regression")
plt.legend()
plt.show()

