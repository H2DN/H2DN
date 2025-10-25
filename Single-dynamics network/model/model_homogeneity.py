import torch
import torch.nn as nn
from model.L0_matrix_dot import L0_dot
from model.L0_matrix_multiple import L0_multiple

class HomoModel(nn.Module):
    def __init__(self, num_categories, num_nodes, num_funcs, device, lamba=8000):
        super(HomoModel, self).__init__()
        self.num_categories = num_categories
        self.num_nodes = num_nodes
        self.num_funcs = num_funcs
        self.device = device
        self.lamba = lamba
        self.N = 50000

        # 同质性：矩阵相乘
        self.homogeneity = L0_multiple(self.num_categories, self.num_funcs, droprate_init=0.5, weight_decay=0,
                        lamba=self.lamba, local_rep=False, temperature=2. / 3., bias=False).to(device)

    def regularization(self):
        regularization = 0.
        for layer in [self.homogeneity]:
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
        output1, weights_homogeneity = self.homogeneity(R)
        output1 = output1 * x  # batch_size, num_nodes, num_funcs
        output1 = torch.sum(output1, dim=-1)
        return output1, weights_homogeneity