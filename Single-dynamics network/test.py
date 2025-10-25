import time, os, torch
import numpy as np
from model.model_homogeneity import HomoModel
from model.model_heterogeneity import HeteModel
from data.data_loader import get_data
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

def test_homogeneity(dxdt, lib, group, lamda, path, get_noise=False):
    # 划分数据集
    train_loader, val_loader, test_loader, all_loader = get_data(dxdt, lib, train_ratio=0.5, val_ratio=0.4, test_ratio=0.1,
                                                     batch_size=64)
    group = torch.tensor(group, dtype=torch.float32).to("cuda")
    # 模型保存地址
    device = torch.device("cuda")
    model = HomoModel(num_categories=3, num_nodes=60, num_funcs=30, device=device, lamba=lamda)  # 模型实例化
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint, strict=True)
    # 训练
    model.eval()
    pred_list = []
    true_list = []

    for item in enumerate(all_loader):
        i_batch, data = item
        trainX = data[0]
        trainY = data[1]  # batch_size, num_station
        predY, homogeneity_param = model(trainX, group)
        true_list.append(trainY.detach().cpu().numpy())
        pred_list.append(predY.detach().cpu().numpy())

    pred = np.concatenate(pred_list, axis=0)
    true = np.concatenate(true_list, axis=0)
    #
    for i in range(60):  #
        # 对比进站客流的真实和预测值
        plt.figure(figsize=(20, 5))
        plt.grid(True)
        plt.plot(pred[:, i])
        plt.plot(true[:, i])
        # plt.plot(pred[:, i]-true[:, i])
        plt.legend(['pred', 'true'])
        path = './stage1_result_figure/' + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        figname_up = path + 'country' + str(i) + '.jpg'
        plt.savefig(figname_up)
        plt.clf()  # 清除当前图
    return homogeneity_param.detach().cpu().numpy()  # num_var, timesteps

def test_heterogeneity(dxdt, lib, group, homogeneity_param, lamda, path, get_noise=False):
    # 划分数据集
    train_loader, val_loader, test_loader, all_loader = get_data(dxdt, lib, train_ratio=0.5, val_ratio=0.4, test_ratio=0.1,
                                                     batch_size=64)
    group = torch.tensor(group, dtype=torch.float32).to("cuda")
    # 模型保存地址
    device = torch.device("cuda")
    homogeneity_param = torch.tensor(homogeneity_param, dtype=torch.float32).to("cuda")
    model = HeteModel(num_categories=3, num_nodes=60, num_funcs=30,  homogeneity_param= homogeneity_param, device=device, lamda=lamda)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint, strict=True)
    # 训练
    model.eval()
    pred_list = []
    true_list = []

    for item in enumerate(all_loader):
        i_batch, data = item
        trainX = data[0]
        trainY = data[1]  # batch_size, num_station
        predY, heterogeneity_param = model(trainX, group)
        true_list.append(trainY.detach().cpu().numpy())
        pred_list.append(predY.detach().cpu().numpy())

    pred = np.concatenate(pred_list, axis=0)  # time_step, num_station
    true = np.concatenate(true_list, axis=0)  # time_step, num_station

    for i in range(60):  #
        # 对比进站客流的真实和预测值
        plt.figure(figsize=(20, 5))
        plt.grid(True)
        plt.plot(pred[:, i])
        plt.plot(true[:, i])
        # plt.plot(pred[:, i]-true[:, i])
        plt.legend(['pred', 'true'])
        path = './stage2_result_figure/' + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        figname_up = path + 'country' + str(i) + '.jpg'
        plt.savefig(figname_up)
        plt.clf()  # 清除当前图
    return heterogeneity_param.detach().cpu().numpy()  # num_var, timesteps






