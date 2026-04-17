import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
data = {
    'Day': [1,2,3,4,5,6,7,8,9,10],
    'Ad_Spend': [100,150,120,130,170,160,180,200,210,190],
    'Discount': [10,15,12,10,20,18,22,25,30,28],
    'Sales': [200,250,220,240,300,280,320,350,370,340]
}

df=pd.DataFrame(data)
print("dataset: ",df)
X=df[['Ad_Spend','Discount']]
y=df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
mae=mean_absolute_error(y_test,y_pred)
print("Mean Absolute Error: ",mae)
ad_spend=int(input("enter the ad spend: "))
discount=int(input("enter the discount: "))
prediction=model.predict([[ad_spend,discount]])
print(f"predicted sales: {prediction[0]:.2f}")
plt.scatter(df['Ad_Spend'], df['Sales'], label="Actual Data")

# Sort values for smooth line
sorted_df = df.sort_values(by='Ad_Spend')

plt.plot(sorted_df['Ad_Spend'], 
         model.predict(sorted_df[['Ad_Spend','Discount']]), 
         label="Regression Line")
plt.scatter(ad_spend,prediction[0],label="Predicted Data")
plt.xlabel("Ad Spend")
plt.ylabel("Sales")
plt.title("Sales Prediction")
plt.legend()
plt.show()
