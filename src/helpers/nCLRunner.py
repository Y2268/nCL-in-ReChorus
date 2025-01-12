import torch
import numpy as np
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, List
import torch.nn.functional as F
from sklearn.cluster import KMeans
from utils import utils
from models.BaseModel import BaseModel
from helpers.BaseRunner import BaseRunner
import ot
class nCLRunner(BaseRunner):
    @staticmethod
    def parse_runner_args(parser):
        parser = BaseRunner.parse_runner_args(parser)
        # parser.add_argument('--k', type=int, default=10, help='The number of clusters.')
        parser.add_argument('--epsilon', type=float, default=1e-3, help='The parameter of the rate distortion function.')
        parser.add_argument('--iopt_iterations', type=int, default=10, help='The number of iterations of IOPT.')
        return parser
    
    def __init__(self, args):
        super(nCLRunner, self).__init__(args)
        self.K = args.k  # 从命令行参数中获取簇数量
        self.iopt_iterations = args.iopt_iterations  # IOPT 迭代次数
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 确定设备

    def fit(self, dataset: BaseModel.Dataset, epoch=-1) -> float:
        model = dataset.model
        model.to(self.device)  # 确保模型在 GPU 上
        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)
        
        model.train()
        loss_lst = list()
        dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                        collate_fn=dataset.collate_batch, pin_memory=True)  # 启用 pin_memory
        for batch in tqdm(dl, leave=False, desc='Epoch {:<3}'.format(epoch), ncols=100, mininterval=1):
            batch = utils.batch_to_gpu(batch, self.device)  # 将 batch 移动到 GPU 上
            
            # 使用 LightGCN 计算用户和物品的嵌入
            out_dict = model.forward(batch)
            user_embeddings = out_dict['u_v'].to(self.device)
            item_embeddings = out_dict['i_v'].to(self.device)
            
            # 归一化嵌入向量
            user_embeddings = F.normalize(user_embeddings, p=2, dim=1)
            item_embeddings = F.normalize(item_embeddings, p=2, dim=1)
        # 更新会员资格矩阵
            model.all_user_membership = self.update_membership_matrix(model,user_embeddings)
            model.all_item_membership = self.update_membership_matrix(model,item_embeddings)
            # 计算损失函数
            loss = model.loss(user_embeddings, item_embeddings,model.all_user_membership,model.all_item_membership)
            
            # 使用 SGD 更新模型参数
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
                
            loss_lst.append(loss.item())
                # 在每个 epoch 结束后，整合所有批次的会员矩阵
        
                # 更新模型中的会员矩阵
                # self.update_membership_matrices(model, dataset)
            # 更新用户的会员矩阵
        return np.mean(loss_lst)
    
    def update_membership_matrix(self,model,embeddings):
                    # 预测用户和物品的簇标签
        cluster_preds = model.mlp(embeddings)
        # 使用softmax函数获取每个嵌入属于每个簇的概率分布
        probabilities = F.softmax(cluster_preds, dim=1)
        # 将概率分布作为会员资格矩阵返回
        return probabilities

###IPOT算法不支持torch.tensor，需要转移到cpu上，如果数据量大，可能会导致内存不足。我们使用了上面更新成员资格矩阵的方法
    # def update_membership_matrix(self, X, num_clusters, max_iter=10, epsilon=1e-3):
    #     """
    #     使用 IPOT 算法更新会员矩阵。
        
    #     参数:
    #     X -- 用户或物品的嵌入向量，形状为 (n_samples, n_features)
    #     num_clusters -- 聚类数量
    #     max_iter -- IPOT 算法的最大迭代次数
    #     epsilon -- IPOT 算法中的正则化参数
        
    #     返回:
    #     membership_matrix -- 会员矩阵，形状为 (n_samples, num_clusters)
    #     """
    #     # 计算成本矩阵
    #     C = ot.dist(X, X)
        
    #     # 使用 IPOT 算法计算最优矩阵 Q*
    #     Q = ot.ipot(X, X, num_clusters, method=' Synder', reg=epsilon, iters=max_iter)
        
    #     # 将 Q 转换为会员矩阵
    #     membership_matrix = np.exp(-C / epsilon) @ Q
        
    #     # 归一化会员矩阵，使得每行的和为 1
    #     membership_matrix /= membership_matrix.sum(axis=1, keepdims=True)
        
    #     return membership_matrix