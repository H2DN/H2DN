import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# single-dynamics network
group_param_dict = {
    0: {"a": 2, "b": 1, "c": 1, "d": 0.5, "e": 0.3},
    1: {"a": 1.5, "b": 0.8, "c": 0.8, "d": 0.4, "e": 0.3},
    2: {"a": 3, "b": 1.5, "c": 1.2, "d": 0.6, "e": 0.3},
}

# Generate network topology through SBM
def build_network():
    sizes = [20, 20, 20]
    probs = [[0.4, 0.05, 0.02],
             [0.05, 0.3, 0.03],
             [0.02, 0.03, 0.25]]
    G = nx.stochastic_block_model(sizes, probs)
    group_ids = []
    for gid, size in enumerate(sizes):
        group_ids.extend([gid] * size)
    return G, group_ids

# 节点类（仅计算导数）
class ComplexNode:
    def __init__(self, group_id, node_id):
        self.gid = group_id
        self.iid = node_id
        self.state = np.random.normal(0.5, 0.1, size=2)  # 二维初始状态
        self.params = self.init_params()  #

    def init_params(self):
        group_params = group_param_dict[self.gid]  # 对应节点的同质性系数
        noise = np.random.normal(0, 0.1, size=2)  # 对应节点的异质性系数
        return {**group_params, "noise": noise}

    def compute_derivative(self, state, neighbor_states):
        x, y = state
        if self.gid == 0:
            xj = np.mean([xn[0] for xn in neighbor_states]) if neighbor_states else 0
            # yj = np.mean([xn[1] for xn in neighbor_states]) if neighbor_states else 0
            yj=0
            dx = self.params['a'] * x - self.params['b'] * x * y + self.params['noise'][0] * x + self.params[
                'e'] * xj
            dy = -self.params['c'] * y + self.params['d'] * x * y + self.params['noise'][1] * y + self.params[
                'e'] * yj
        elif self.gid == 1:
            xj = np.mean([xn[0] for xn in neighbor_states]) if neighbor_states else 0
            # yj = np.mean([xn[1] for xn in neighbor_states]) if neighbor_states else 0
            yj = 0
            dx = self.params['a'] * x - self.params['b'] * x * y + self.params['noise'][0] * x + self.params[
                'e'] * xj
            dy = -self.params['c'] * y + self.params['d'] * x * y + self.params['noise'][1] * y + self.params[
                'e'] * yj
        elif self.gid == 2:
            xj = np.mean([xn[0] for xn in neighbor_states]) if neighbor_states else 0
            # yj = np.mean([xn[1] for xn in neighbor_states]) if neighbor_states else 0
            yj = 0
            dx = self.params['a'] * x - self.params['b'] * x * y + self.params['noise'][0] * x + self.params[
                'e'] * xj
            dy = -self.params['c'] * y + self.params['d'] * x * y + self.params['noise'][1] * y + self.params[
                'e'] * yj
        return np.array([dx, dy])

# Runge-Kutta 单步积分
def rk4_step(node, state, neighbor_states_func, dt):
    k1 = node.compute_derivative(state, neighbor_states_func(state))
    k2 = node.compute_derivative(state + 0.5 * dt * k1, neighbor_states_func(state + 0.5 * dt * k1))
    k3 = node.compute_derivative(state + 0.5 * dt * k2, neighbor_states_func(state + 0.5 * dt * k2))
    k4 = node.compute_derivative(state + dt * k3, neighbor_states_func(state + dt * k3))
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4), k1

# 主仿真函数（用RK4）
def simulate(G, group_ids, T=10, dt=0.01):
    nodes = [ComplexNode(group_ids[i], i) for i in range(len(G))]  # 节点实例化

    history = []
    derivative_history = []  # 保存所有节点的导数（dx/dt 和 dy/dt）
    init_states = np.array([node.state.copy() for node in nodes])
    noise = np.array([node.params['noise'].copy() for node in nodes])

    for t in range(int(T / dt)):
        current_states = [node.state.copy() for node in nodes]  # 当前状态(nodes, dx dy), 60, 2

        # 构造邻居状态查询函数（动态更新）
        def make_neighbor_func(i):
            neighbors = list(G.neighbors(i))
            return lambda x: [current_states[j] for j in neighbors]

        # RK4 更新每个节点状态
        derivatives_at_t = []
        for i, node in enumerate(nodes):
            neighbor_states_func = make_neighbor_func(i)  # 返回邻居节点的当前状态
            node.state, k1 = rk4_step(node, current_states[i], neighbor_states_func, dt)
            derivatives_at_t.append(k1)

        history.append(np.array([node.state.copy() for node in nodes]))
        derivative_history.append(np.array(derivatives_at_t))  # 保存所有节点的导数
    return np.stack(history), np.stack(derivative_history), init_states, noise

# 可视化函数
def plot_trajectory(history, node_idx):
    traj = history[:, node_idx, 0]
    plt.plot(traj)
    plt.title(f"Node {node_idx} State Trajectory (x)")
    plt.xlabel("Time Step")
    plt.ylabel("State")
    plt.show()

# 主程序入口
if __name__ == '__main__':
    G, group_ids = build_network()
    A = pd.DataFrame(nx.to_numpy_array(G))
    A.to_csv('adj.csv', index=False, header=False)
    sim_data, derivative_data, init_states, noise = simulate(G, group_ids, T=30, dt=0.01)
    print(derivative_data.shape)
    print(init_states.shape)
    print(noise.shape)
    x = pd.DataFrame(sim_data[:, :, 0])
    y = pd.DataFrame(sim_data[:, :, 1])
    dx = pd.DataFrame(derivative_data[:, :, 0])
    dy = pd.DataFrame(derivative_data[:, :, 1])
    inti_states = pd.DataFrame(init_states)
    noise = pd.DataFrame(noise)
    x.to_csv('x.csv', index=False, header=False)  # x dimension node state
    y.to_csv('y.csv',  index=False, header=False)  # y dimension node state
    dx.to_csv('dxdt.csv', index=False, header=False)  # x dimension dynamics state
    dy.to_csv('dydt.csv',  index=False, header=False)  # y dimension dynamics state
    inti_states.to_csv('inti_states.csv',  index=False, header=False)  # initial state of the system
    noise.to_csv('heterogeneity_param.csv',  index=False, header=False)  # Heterogeneity parameters
    plot_trajectory(derivative_data, node_idx=37)  # Visualization
