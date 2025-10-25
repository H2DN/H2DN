import torch
import torch.nn as nn
from model.L0_matrix_dot import L0_dot
from model.L0_matrix_multiple import L0_multiple


class HeteModel(nn.Module):
    def __init__(self, num_categories, num_nodes, num_funcs, homogeneity_param, device, lamda):
        super(HeteModel, self).__init__()
        self.num_categories = num_categories
        self.num_nodes = num_nodes
        self.num_funcs = num_funcs
        self.homogeneity_param = homogeneity_param
        self.device = device
        self.lamda = lamda
        self.N = 50000

        # 异质性：矩阵点积
        self.heterogeneity = L0_dot(self.num_nodes, self.num_funcs, droprate_init=0.5, weight_decay=0,
                                    lamba=self.lamda, local_rep=False, temperature=2. / 3., bias=False).to(device)

    def regularization(self):
        regularization = 0.
        for layer in [self.heterogeneity]:
            regularization += - (1. / self.N) * layer.regularization()
        if torch.cuda.is_available():
            regularization = regularization.cuda()
        return regularization

    def forward(self, x, R):
        '''
        :param x: 函数库数据, batch_size, num_nodes, num_funcs
        :param R: 节点关联矩阵, num_nodes, num_categories
        :return:
        '''

        # 同质性
        output1 = torch.matmul(R, self.homogeneity_param)
        output1 = output1 * x  # batch_size, num_nodes, num_funcs
        output1 = torch.sum(output1, dim=-1)

        # 异质性
        output2, weights_heterogeneity = self.heterogeneity(x)
        output2 = torch.sum(output2, dim=-1)

        return output1 + output2, weights_heterogeneity
