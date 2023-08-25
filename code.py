import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


data = pd.read_csv("CC.csv")


data.drop("CUST_ID", axis=1, inplace=True)
data.dropna(inplace=True)


data.hist(figsize=(15,15))
plt.show()

sns.heatmap(data.corr(), annot=True, cmap="Blues")
plt.show()


scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)


ssd = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(scaled_data)
    ssd.append(kmeans.inertia_)

plt.plot(range(1, 11), ssd, marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("Sum of squared distances")
plt.show()


k = 4  
kmeans = KMeans(n_clusters=k, random_state=0)
clusters = kmeans.fit_predict(scaled_data)
data["Cluster"] = clusters


pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
data["PCA1"] = pca_result[:, 0]
data["PCA2"] = pca_result[:, 1]


sns.heatmap(data[["PCA1", "PCA2", "Cluster"]].corr(), annot=True, cmap="Blues")
plt.show()

sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=data)
plt.title('PCA Results')
plt.show()
