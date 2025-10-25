import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fcmeans import FCM
from sklearn.preprocessing import StandardScaler

# 计算 FCM 损失函数值
def fcm_loss(X, centers, u, m):
    loss = 0
    for i in range(X.shape[0]):
        for j in range(centers.shape[0]):
            loss += (u[i, j] ** m) * np.linalg.norm(X[i] - centers[j]) ** 2
    return loss


# 0.导入数据
x = pd.read_csv("../data/temp.csv", header=0, index_col=0, delimiter=',').T
x = np.array(x)[:, :168]

# 1.归一化
scaler = StandardScaler()
x = scaler.fit_transform(x.T).T  # 对每个车站独立标准化

# 2.聚类，采用肘部原则
loss_list = []
for i in range(5, 6):
    # 创建 Fuzzy C-Means 模型
    fcm = FCM(n_clusters=i, m=2, max_iter=300, error=1e-5, random_state=42)
    # 拟合模型
    fcm.fit(x)
    # 获取聚类中心
    centers = fcm.centers
    # 获取每个数据点的隶属度
    u = fcm.u
    # 构建聚类标签矩阵 (样本数，类别数)
    cluster_matrix = np.zeros(u.shape)
    # 设定最大隶属度为 1 的类别为该样本的类别
    for idx in range(u.shape[0]):
        cluster_matrix[idx, np.argmax(u[idx])] = 1

    # 保存聚类标签矩阵到CSV
    cluster_matrix_df = pd.DataFrame(cluster_matrix)
    cluster_matrix_df.to_csv('../data/group.csv', index=False, header=False)
    loss_sin = fcm_loss(x, centers, u, m=2)
    loss_list.append(loss_sin)
    print(f"FCM Loss: {loss_sin}")