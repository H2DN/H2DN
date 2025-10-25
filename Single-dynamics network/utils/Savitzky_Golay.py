import numpy as np
from scipy.signal import savgol_filter

def Savitzky_Golay(x, t):
    '''
    Savitzky-Golay Numerical Differentiation
    :param x: network dynamics, (num_var, timesteps)
    :param t: time stamp
    :return: dx_dt, numerical derivative of x with respect to t
    '''
    # 使用 Savitzky-Golay 滤波进行平滑
    window_length = 11  # 窗口大小，必须是奇数
    polyorder = 3  # 多项式阶数

    # 对所有变量进行平滑和求导
    dx_dt = np.zeros_like(x)  # 存储一阶导数
    for i in range(len(x)):
        dx_dt[i, :] = savgol_filter(x[i, :], window_length, polyorder, deriv=1, delta=t[1] - t[0])
    return dx_dt

def five_point_derivative(x, t):
    '''
    五点求导法数值微分
    :param x: 多变量时间序列，形状 (num_vars, timesteps)
    :param t: 时间戳（需等间隔）
    :return: dx_dt，数值导数
    '''
    dx_dt = np.zeros_like(x)
    dt = t[1] - t[0]  # 假设时间等间隔
    num_vars, timesteps = x.shape

    # 核验时间均匀性（可选）
    if not np.allclose(np.diff(t), dt):
        raise ValueError("时间戳必须等间隔")

    for i in range(num_vars):
        signal = x[i, :]
        derivative = np.zeros(timesteps)

        # 边界点处理
        derivative[0] = (-3 * signal[0] + 4 * signal[1] - signal[2]) / (2 * dt)
        derivative[1] = (signal[2] - signal[0]) / (2 * dt)
        derivative[-2] = (signal[-1] - signal[-3]) / (2 * dt)
        derivative[-1] = (3 * signal[-1] - 4 * signal[-2] + signal[-3]) / (2 * dt)

        # 内部点使用五点中心差分
        for j in range(2, timesteps - 2):
            derivative[j] = (-signal[j + 2] + 8 * signal[j + 1] - 8 * signal[j - 1] + signal[j - 2]) / (12 * dt)

        dx_dt[i, :] = derivative

    return dx_dt