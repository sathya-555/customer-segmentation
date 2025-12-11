# -------------------------------------------
# Customer Segmentation Using K-Means
# -------------------------------------------

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------- Step 1: Load Sample Data ----------
data = {
    'Age': [25, 34, 22, 45, 52, 23, 40, 30, 36, 48],
    'Income': [30000, 52000, 28000, 80000, 95000, 26000, 60000, 45000, 50000, 85000],
    'Frequency': [2, 6, 1, 8, 7, 1, 5, 3, 4, 6],
    'Spending': [200, 400, 150, 700, 800, 100, 350, 280, 300, 650]
}

df = pd.DataFrame(data)
print("Sample Data:")
print(df)

# ---------- Step 2: Scale the Data ----------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# ---------- Step 3: Choose K (No. of Clusters) ----------
k = 4  # You can change this to 3, 4, or 5
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(scaled_data)
df['Cluster'] = clusters

print("\nClustered Data:")
print(df)

# ---------- Step 4: 2D Visualization ----------
plt.figure(figsize=(8, 6))
plt.scatter(df['Income'], df['Spending'], c=df['Cluster'])
plt.xlabel("Income")
plt.ylabel("Spending")
plt.title("Customer Segmentation (Income vs Spending)")
plt.grid(True)
plt.show()

# ---------- Step 5: 3D Visualization ----------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df['Age'], df['Income'], df['Spending'], c=df['Cluster'], s=60)
ax.set_xlabel("Age")
ax.set_ylabel("Income")
ax.set_zlabel("Spending")
ax.set_title("Customer Segmentation (3D View)")
plt.show()
