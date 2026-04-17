import pandas as pd 
import matplotlib.pyplot as plt
data = {
    'Ad_Spend': [100,150,120,130,170,160,180,200,210,190],
    'Discount': [10,15,12,10,20,18,22,25,30,28],
    'Sales': [200,250,220,240,300,280,320,350,370,340]
}

df=pd.DataFrame(data)
print("dataset: ",df)
corr=df.corr()
print("\nCorrelation matrix:\n",corr)
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns)
plt.yticks(range(len(corr)), corr.columns)

plt.title("Correlation Matrix Heatmap")
plt.show()
