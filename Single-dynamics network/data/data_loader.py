import torch
from torch.utils.data import TensorDataset, random_split, DataLoader, Subset


def get_data(dxdt, lib, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, batch_size=256):
    # 划分数据集
    # 转换为 PyTorch Tensor
    dxdt = torch.tensor(dxdt, dtype=torch.float32).to("cuda")
    lib = torch.tensor(lib, dtype=torch.float32).to("cuda")
    # 构造 PyTorch Dataset
    dataset = TensorDataset(lib, dxdt)
    # 数据集大小
    n_samples = len(dataset)
    # 计算各个数据集的大小
    train_size = int(train_ratio * n_samples)
    val_size = int(val_ratio * n_samples)
    test_size = n_samples - train_size - val_size  # 确保总大小一致

    # 按索引顺序划分数据集
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, train_size + val_size + test_size))

    # 使用 Subset 创建子数据集
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # 使用 DataLoader 为每个数据集创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    all_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, all_loader
