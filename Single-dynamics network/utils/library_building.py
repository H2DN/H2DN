import numpy as np

def five_step_moving_average_fast(x):
    '''
    使用滑动窗口计算历史五步平均
    :param x: 多变量时间序列，形状 (num_vars, timesteps)
    :return: five_step_avg，形状 (num_vars, timesteps)
    '''
    kernel = np.ones(5) / 5  # 归一化的滑动窗口核
    num_vars, timesteps = x.shape
    five_step_avg = np.zeros_like(x, dtype=np.float64)

    for i in range(num_vars):
        five_step_avg[i, :] = np.convolve(x[i, :], kernel, mode='full')[:timesteps]
    return five_step_avg

def  library_building(x, y, adj):
    lib = {}

    # 常数项
    lib_ele = np.ones_like(x)
    lib['1'] = lib_ele

    # x
    lib_ele = x
    lib['x'] = lib_ele

    # x^2
    lib_ele = x ** 2
    lib['x^2'] = lib_ele

    # x^3
    lib_ele = x ** 3
    lib['x^3'] = lib_ele

    # y
    lib_ele = y
    lib['y'] = lib_ele

    # y^2
    lib_ele = y ** 2
    lib['y^2'] = lib_ele

    # y^3
    lib_ele = y ** 3
    lib['y^3'] = lib_ele

    # sin(x)
    lib_ele = np.sin(x)
    lib['sin(x)'] = lib_ele

    # cos(x)
    lib_ele = np.cos(x)
    lib['cos(x)'] = lib_ele

    # tan(x)
    lib_ele = np.tan(x)
    lib['tan(x)'] = lib_ele

    # sin(y)
    lib_ele = np.sin(y)
    lib['sin(y)'] = lib_ele


    # tan(y)
    lib_ele = np.tan(y)
    lib['tan(y)'] = lib_ele

    # xy
    lib_ele = x * y
    lib['xy'] = lib_ele

    # xxy
    lib_ele = x * x * y
    lib['xxy'] = lib_ele

    # xsin(x)
    lib_ele = x * np.sin(x)
    lib['xsin(x)'] = lib_ele

    # xcos(x)
    lib_ele = x * np.cos(x)
    lib['xcos(x)'] = lib_ele

    # ysin(y)
    lib_ele = y * np.sin(y)
    lib['ysin(y)'] = lib_ele

    # ycos(y)
    lib_ele = y * np.cos(y)
    lib['ycos(y)'] = lib_ele

    # ex
    lib_ele = np.exp(x)
    lib['exp(x)'] = lib_ele

    # ey
    lib_ele = np.exp(y)
    lib['exp(y)'] = lib_ele

    # xj
    lib_ele = np.zeros_like(x)
    for i in range(len(x)):
        neighbor = adj[i, :]
        neighbor = neighbor[:, np.newaxis]
        ele = x * neighbor
        lib_ele[i] = np.sum(ele, axis=0) / np.sum(adj[i, :])
    lib['xj'] = lib_ele

    # yj
    lib_ele = np.zeros_like(x)
    for i in range(len(x)):
        neighbor = adj[i, :]
        neighbor = neighbor[:, np.newaxis]
        ele = y * neighbor
        lib_ele[i] = np.sum(ele, axis=0) / np.sum(adj[i, :])
    lib['yj'] = lib_ele

    # sin(xi - xj)
    lib_ele = np.zeros_like(x)
    for i in range(len(x)):
        neighbor = adj[i, :]
        neighbor = neighbor[:, np.newaxis]
        ele = np.expand_dims(x[i, :], axis=0) * neighbor - x * neighbor
        lib_ele[i] = np.sum(np.sin(ele), axis=0) / np.sum(adj[i, :])
    lib['sin(xi-xj)'] = lib_ele

    # sin(yi - yj)
    lib_ele = np.zeros_like(x)
    for i in range(len(x)):
        neighbor = adj[i, :]
        neighbor = neighbor[:, np.newaxis]
        ele = np.expand_dims(y[i, :], axis=0) * neighbor - y * neighbor
        lib_ele[i] = np.sum(np.sin(ele), axis=0) / np.sum(adj[i, :])
    lib['sin(yi-yj)'] = lib_ele

    # cos(xi - xj)
    lib_ele = np.zeros_like(x)
    for i in range(len(x)):
        neighbor = adj[i, :]
        neighbor = neighbor[:, np.newaxis]
        ele = np.expand_dims(x[i, :], axis=0) * neighbor - x * neighbor
        lib_ele[i] = np.sum(np.cos(ele), axis=0) / np.sum(adj[i, :])
    lib['cos(xi-xj)'] = lib_ele

    # cos(yi - yj)
    lib_ele = np.zeros_like(x)
    for i in range(len(x)):
        neighbor = adj[i, :]
        neighbor = neighbor[:, np.newaxis]
        ele = np.expand_dims(y[i, :], axis=0) * neighbor - y * neighbor
        lib_ele[i] = np.sum(np.cos(ele), axis=0) / np.sum(adj[i, :])
    lib['cos(yi-yj)'] = lib_ele

    # sin(xj)
    lib_ele = np.zeros_like(x)
    for i in range(len(x)):
        neighbor = adj[i, :]
        neighbor = neighbor[:, np.newaxis]
        ele = np.sum(np.sin(x * neighbor), axis=0)
        lib_ele[i] = ele
    lib['sin(xj)'] = lib_ele

    # cos(xj)
    lib_ele = np.zeros_like(x)
    for i in range(len(x)):
        neighbor = adj[i, :]
        neighbor = neighbor[:, np.newaxis]
        ele = np.sum(np.cos(x * neighbor), axis=0)
        lib_ele[i] = ele
    lib['cos(xj)'] = lib_ele

    # sin(yj)
    lib_ele = np.zeros_like(x)
    for i in range(len(x)):
        neighbor = adj[i, :]
        neighbor = neighbor[:, np.newaxis]
        ele = np.sum(np.sin(y * neighbor), axis=0)
        lib_ele[i] = ele
    lib['sin(yj)'] = lib_ele

    # cos(yj)
    lib_ele = np.zeros_like(x)
    for i in range(len(x)):
        neighbor = adj[i, :]
        neighbor = neighbor[:, np.newaxis]
        ele = np.sum(np.cos(y * neighbor), axis=0)
        lib_ele[i] = ele
    lib['cos(yj)'] = lib_ele

    return lib