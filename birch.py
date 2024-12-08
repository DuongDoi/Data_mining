import pandas as pd
from sklearn.cluster import Birch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Đọc dữ liệu
df = pd.read_csv("lris.csv")
data = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].dropna().values

# Cấu hình BIRCH
birch_model = Birch(n_clusters=3, threshold=0.5)  # threshold điều khiển kích thước cụm
birch_model.fit(data)

# Lấy nhãn cụm
labels = birch_model.labels_
centroids = birch_model.subcluster_centers_

# Giảm chiều dữ liệu xuống 2D bằng PCA
pca = PCA(n_components=2)
data_2d = pca.fit_transform(data)

# Vẽ biểu đồ
plt.figure(figsize=(10, 7))
colors = ['red', 'blue', 'green']

for cluster_id in set(labels):
    cluster_points = data_2d[labels == cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_id+1}', color=colors[cluster_id])

# Vẽ centroids
centroids_2d = pca.transform(centroids)
plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], s=200, c='yellow', marker='X', label='Centroids')

plt.title("BIRCH Clustering")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.show()
