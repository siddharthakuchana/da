import pandas as pd
import matplotlib.pyplot as plt
data = {
    'Age': [25, 45, 35, 50, 23, 40, 60, 55, 30, 48],
    'BP': [120, 140, 130, 150, 110, 135, 160, 155, 125, 145],
    'Cholesterol': [200, 240, 220, 260, 180, 230, 270, 250, 210, 245],
    'Disease': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes']
}

df=pd.DataFrame(data)
print("dataset: ",df)
print("\nStatistical Summary:\n")
print(df.describe())
print("\nAverage values based on Disease:\n")
print(df.groupby('Disease').mean())
plt.hist(df['Age'])
plt.xlabel("age")
plt.ylabel("number of patients")
plt.title("age distribution")
plt.show()

plt.scatter(df['BP'], df['Cholesterol'])
plt.xlabel("blood pressure")
plt.ylabel("cholesterol")
plt.show()

disease_counts = df['Disease'].value_counts()
disease_counts.plot(kind='bar')
plt.xlabel("Disease (Yes/No)")
plt.ylabel("Count")
plt.title("Disease Distribution")
plt.show()
