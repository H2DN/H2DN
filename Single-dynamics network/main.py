import numpy as np
import pandas as pd
from utils.Savitzky_Golay import Savitzky_Golay
from utils.library_building import library_building
from train import train_homogeneity, train_heterogeneity
from test import test_homogeneity, test_heterogeneity

## 1.load data
x_origin = np.array(pd.read_csv('./data/x.csv', header=None, index_col=None, delimiter=',')).T  # num_nodes, num_timesteps
y_origin = np.array(pd.read_csv('./data/y.csv', header=None, index_col=None, delimiter=',')).T  # num_nodes, num_timesteps
adj = np.array(pd.read_csv('./data/adj.csv', header=None, index_col=None, delimiter=','))  # num_nodes, num_nodes
tspan = np.arange(0, 30, 0.01)  # range_time
group = np.array(pd.read_csv('./data/group.csv', header=None, index_col=None))  # num_nodes, num_groups
print('1')

## 2.Numerical Differentiation
dxdt = Savitzky_Golay(x_origin, tspan)  # num_nodes, num_timesteps
dydt = Savitzky_Golay(y_origin, tspan)  # num_nodes, num_timesteps

## 3.Building function library
lib = library_building(x_origin, y_origin, adj)
for key, value in lib.items():
    # 检查值是否为二维 ndarray
    if isinstance(value, np.ndarray) and value.ndim == 2:
        print(value.shape)
        # 将 ndarray 转换为 DataFrame
        df = pd.DataFrame(value)
        # 保存为 CSV 文件，文件名使用字典的键
        filename = f"./library/{key}.csv"
        df.to_csv(filename, index=False, header=False)
        print(f"Saved {filename}")
    else:
        print(f"Value associated with {key} is not a 2D ndarray.")
lib = np.stack(list(lib.values()), axis=2)  # num_nodes, num_timesteps, num_funcs
print(lib.shape)
print('3')



## 4.Stage1: Inferring homogeneity dynamics
# 训练
# train_homogeneity(dxdt.swapaxes(0, 1), lib.swapaxes(0, 1), group, 5000, lr=0.001, epochs=10000, model_type='homogeneity_x')  # x-dimension
# train_homogeneity(dydt.swapaxes(0, 1), lib.swapaxes(0, 1), group, 8000, lr=0.001, epochs=10000, model_type='homogeneity_y')  # y-dimension
# 测试
# homogeneity_param = test_homogeneity(dxdt.swapaxes(0, 1), lib.swapaxes(0, 1), group, 8000, r"D:\Paper2\文件整理\code\single-dynamics network\save_model\5000-x_2025_06_20_17_02_02\model_dict_checkpoint_9989_0.94236493.pth")
# homogeneity_param = pd.DataFrame(homogeneity_param)
# homogeneity_param.to_csv("./results/homogeneity_param_x.csv", index=False, header=False)
# homogeneity_param = test_homogeneity(dydt.swapaxes(0, 1), lib.swapaxes(0, 1), group, 8000, r"D:\Paper2\文件整理\code\single-dynamics network\save_model\8000-y_2025_06_20_17_05_57\model_dict_checkpoint_9753_1.02781899.pth")
# homogeneity_param = pd.DataFrame(homogeneity_param)
# homogeneity_param.to_csv("./results/homogeneity_param_y.csv", index=False, header=False)

## 5.Stage2: Inferring heterogeneity dynamics
# 训练
# homogeneity_param_x = np.array(pd.read_csv('./results/homogeneity_param_x.csv', header=None, index_col=None, delimiter=','))
# train_heterogeneity(dxdt.swapaxes(0, 1), lib.swapaxes(0, 1), group, lamda=4.5,homogeneity_param=homogeneity_param_x, lr=0.1, epochs=50000, model_type='heterogeneity_x')  # x-dimension
# homogeneity_param_y = np.array(pd.read_csv('./results/homogeneity_param_y.csv', header=None, index_col=None, delimiter=','))
# train_heterogeneity(dydt.swapaxes(0, 1), lib.swapaxes(0, 1), group, lamda=4.5,homogeneity_param=homogeneity_param_y, lr=0.1, epochs=50000, model_type='heterogeneity_x')  # y-dimension

# 测试
# heterogeneity_param_x = test_heterogeneity(dxdt.swapaxes(0, 1), lib.swapaxes(0, 1), group, homogeneity_param_x, 4.5, r"D:\Paper2\文件整理\code\single-dynamics network\save_model\4.5-yizhi-x_2025_06_22_11_43_06\model_dict_checkpoint_7852_0.00714301.pth")
# heterogeneity_param_x = pd.DataFrame(heterogeneity_param_x)
# heterogeneity_param_x.to_csv("./data/heterogeneity_param_x.csv", index=False, header=False)

# heterogeneity_param_y = test_heterogeneity(dydt.swapaxes(0, 1), lib.swapaxes(0, 1), group, homogeneity_param_y, 4.5, r"D:\Paper2\文件整理\code\single-dynamics network\save_model\4.5-yizhi-y_2025_06_22_11_44_46\model_dict_checkpoint_9119_0.00909849.pth")
# heterogeneity_param_y = pd.DataFrame(heterogeneity_param_y)
# heterogeneity_param_y.to_csv("./data/heterogeneity_param_y.csv", index=False, header=False)

