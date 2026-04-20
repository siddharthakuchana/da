import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.datasets import load_iris

# Load dataset once
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Individual Correlation between each pair
cols = df.columns
print("\nPairwise Correlations:")
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        corr, pval = pearsonr(df[cols[i]], df[cols[j]])
        print(f"{cols[i]} vs {cols[j]} => r = {corr:.4f}, p-value = {pval:.6f}")

# Full Correlation Matrix
print("\nCorrelation Matrix:\n", df.corr().round(4))

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, fmt='.3f', cmap='coolwarm', square=True)
plt.title('Correlation Heatmap - Iris Dataset')
plt.tight_layout()
plt.show()
