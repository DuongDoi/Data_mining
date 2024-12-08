import pandas as pd
from pyclustering.cluster.clarans import clarans
from pyclustering.utils import read_sample
from pyclustering.cluster import cluster_visualizer

# Đọc dữ liệu
df = pd.read_csv("Iris.csv")


data = df[['SepalLengthCm','SepalWidthCm']].head(150).values
#chọn các thuộc tính tương ứng, số lượng thuộc tính là số chiều dữ liệu
num_clusters = 5  # Số cụm
num_local = 5  # Số tìm kiếm ngẫu nhiên
max_neighbor = 10  # Số hàng xóm tối đa trong mỗi tìm kiếm

# Áp dụng CLARANS
clarans_instance = clarans(data, num_clusters, num_local, max_neighbor)
clarans_instance.process()

# Lấy kết quả
clusters = clarans_instance.get_clusters()  # Các cụm
medoids = clarans_instance.get_medoids()  # Medoids
# Trực quan hóa cụm
visualizer = cluster_visualizer()
visualizer.append_clusters(clusters, data)
visualizer.show()