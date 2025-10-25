import time, os, torch
import numpy as np
import torch.nn as nn
from model.model_homogeneity import HomoModel
from model.model_heterogeneity import HeteModel
from utils.earlystopping import EarlyStopping
from data.data_loader import get_data

def train_homogeneity(dxdt, lib, group, lamba, lr=0.001, epochs=2000, model_type='homogeneity'):

    # 划分数据集
    train_loader, val_loader, test_loader, all_loader = get_data(dxdt, lib, train_ratio=0.4, val_ratio=0.5,
                                                                 test_ratio=0.1,
                                                                 batch_size=64)
    group = torch.tensor(group, dtype=torch.float32).to("cuda")

    # 模型保存地址
    device = torch.device("cuda")
    TIMESTAMP = str(time.strftime("%Y_%m_%d_%H_%M_%S"))
    save_dir = './save_model/' + model_type + '_' + TIMESTAMP
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    global_start_time = time.time()
    model = HomoModel(num_categories=3, num_nodes=60, num_funcs=30, device=device, lamba=lamba)  # 模型实例化
    MSELoss = nn.MSELoss()

    def criterion(output, target_var, model):
        loss = MSELoss(output, target_var)
        total_loss = loss + model.regularization()
        if torch.cuda.is_available():
            total_loss = total_loss.cuda()
        return total_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 优化器
    early_stopping = EarlyStopping(patience=200, verbose=True)  # 早停
    temp_time = time.time()
    count = 0
    # 训练
    for epoch in range(epochs):
        count += 1
        train_loss = 0
        model.train()
        for item in enumerate(train_loader):
            i_batch, data = item
            trainX = data[0]
            trainY = data[1]
            predY, _ = model(trainX, group)
            loss = criterion(trainY, predY, model)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            model.eval()
            val_loss = 0
            for item in enumerate(val_loader):
                i_batch, data = item
                valX = data[0]
                valY = data[1]
                predY, _ = model(valX, group)
                loss = criterion(valY, predY, model)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch: {epoch + 1:2} Train_Loss:{avg_train_loss:10.8f} Val_Loss:{avg_val_loss:10.8f}')
        if epoch > 0:
            # early stopping
            model_dict = model.state_dict()
            early_stopping(avg_val_loss, model_dict, model, epoch, save_dir)
            if early_stopping.early_stop:
                print("Early Stopping")
                break
        # 每10个epoch打印一次训练时间
        if epoch % 10 == 0:
            print("time for 10 epoches:", round(time.time() - temp_time, 2))
            temp_time = time.time()
    global_end_time = time.time() - global_start_time
    print("global end time:", global_end_time)


def train_heterogeneity(dxdt, lib, group, lamda, homogeneity_param, lr=0.001, epochs=2000, model_type='heterogeneity'):
    # 划分数据集
    train_loader, val_loader, test_loader, all_loader = get_data(dxdt, lib, train_ratio=0.4, val_ratio=0.5,
                                                                 test_ratio=0.1,
                                                                 batch_size=64)
    group = torch.tensor(group, dtype=torch.float32).to("cuda")

    # 模型保存地址
    device = torch.device("cuda")
    TIMESTAMP = str(time.strftime("%Y_%m_%d_%H_%M_%S"))
    save_dir = './save_model/' + model_type + '_' + TIMESTAMP
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    global_start_time = time.time()
    homogeneity_param = torch.tensor(homogeneity_param, dtype=torch.float32).to("cuda")
    model = HeteModel(num_categories=3, num_nodes=60, num_funcs=30, homogeneity_param=homogeneity_param, device=device, lamda=lamda)  # 模型实例化
    MSELoss = nn.MSELoss()

    def criterion(output, target_var, model):
        loss = MSELoss(output, target_var)
        total_loss = loss + model.regularization()
        if torch.cuda.is_available():
            total_loss = total_loss.cuda()
        return total_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 优化器
    early_stopping = EarlyStopping(patience=200, verbose=True)  # 早停
    temp_time = time.time()
    count = 0
    # 训练
    for epoch in range(epochs):
        count += 1
        train_loss = 0
        model.train()
        for item in enumerate(train_loader):
            i_batch, data = item
            trainX = data[0]
            trainY = data[1]
            predY, _ = model(trainX, group)
            loss = criterion(trainY, predY, model)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            # model validation
            model.eval()
            val_loss = 0
            for item in enumerate(val_loader):
                i_batch, data = item
                valX = data[0]
                valY = data[1]
                predY, _ = model(valX, group)
                loss = criterion(valY, predY, model)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch: {epoch + 1:2} Train_Loss:{avg_train_loss:10.8f} Val_Loss:{avg_val_loss:10.8f}')
        if epoch > 0:
            # early stopping
            model_dict = model.state_dict()
            early_stopping(avg_val_loss, model_dict, model, epoch, save_dir)
            if early_stopping.early_stop:
                print("Early Stopping")
                break
        # 每10个epoch打印一次训练时间
        if epoch % 10 == 0:
            print("time for 10 epoches:", round(time.time() - temp_time, 2))
            temp_time = time.time()
    global_end_time = time.time() - global_start_time
    print("global end time:", global_end_time)
