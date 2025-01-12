import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BaseModel import GeneralModel
from models.general.LightGCN import LightGCNBase
import numpy as np
# from sklearnex import patch_sklearn, unpatch_sklearn
# patch_sklearn()
import torch_kmeans

class nCL(GeneralModel, LightGCNBase):
    reader = 'BaseReader'
    runner = 'nCLRunner'
    extra_log_args = ['emb_size', 'n_layers', 'batch_size']

    @staticmethod
    def parse_model_args(parser):
        # 首先解析 LightGCNBase 模型的参数
        parser = LightGCNBase.parse_model_args(parser)
        
        # 然后解析 GeneralModel 模型的参数
        parser = GeneralModel.parse_model_args(parser)
        
        # 添加新的命令行参数 alpha
        parser.add_argument('--alpha', type=float, default=1.0, help='The balance parameter between alignment and compactness loss.')
        # 添加新的命令行参数 簇数量K
        parser.add_argument('--k', type=int, default=10, help='The number of clusters.')
        return parser
    
    def __init__(self, args, corpus):
        GeneralModel.__init__(self, args, corpus)
        self._base_init(args, corpus)
        self.alpha = args.alpha  # 以命令行的方式获得紧凑性损失的权重
        self.K = args.k  # 以命令行的方式获得簇数量
        self.user_embeddings = None
        self.item_embeddings = None
        self.all_user_membership = None
        self.all_item_membership = None
        # 初始化MLP聚类模型
        self.mlp = nn.Sequential(
            nn.Linear(args.emb_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.K),
            nn.Softmax(dim=1)
        )
        
        # 将MLP聚类模型移动到GPU
        self.mlp.to('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, feed_dict):
        out_dict = LightGCNBase.forward(self, feed_dict)  # 调用父类的forward方法
        if out_dict is None:
            raise ValueError("out_dict is None. Please check the implementation of LightGCNBase.forward.")
        
        # 存储用户和物品嵌入
        self.user_embeddings = out_dict['u_v']
        self.item_embeddings = out_dict['i_v']

        return {'prediction': out_dict['prediction'], 'u_v': out_dict['u_v'], 'i_v': out_dict['i_v']}

    def loss(self, user_embeddings, item_embeddings,user_membership, item_membership):
        # 实现对齐损失和紧凑性损失
        delta = 0.1
        user_compactness_loss = self.compactness_loss(user_embeddings,user_membership, delta)
        item_compactness_loss = self.compactness_loss(item_embeddings, item_membership,delta)
        alignment_loss = torch.mean(torch.sum((user_embeddings - item_embeddings) ** 2, dim=1))
        total_loss = alignment_loss + self.alpha * (user_compactness_loss + item_compactness_loss)
        return total_loss



    def compactness_loss(self, embeddings, membership,delta):
        def rate_distortion_function(embeddings, delta):
            embeddings = embeddings.view(-1, embeddings.shape[-1])
            batch_size = 256
            num_batches = (embeddings.shape[0] + batch_size - 1) // batch_size
            log_det_sum = 0
            for i in range(num_batches):
                batch_embeddings = embeddings[i * batch_size:(i + 1) * batch_size]
                cov_matrix = torch.matmul(batch_embeddings, batch_embeddings.t()) / batch_embeddings.shape[0]
                log_det = torch.logdet(cov_matrix + delta * torch.eye(cov_matrix.shape[0], device=cov_matrix.device))
                log_det_sum += log_det
            return log_det_sum / (2 * num_batches)

        embeddings = embeddings.view(-1, embeddings.shape[-1])
        overall_rate = rate_distortion_function(embeddings, delta)

        cluster_indices = self.deep_clustering(embeddings)
        cluster_rates = []

        for cluster_idx in range(self.K):
            cluster_mask = cluster_indices == cluster_idx
            cluster_embeddings = embeddings[cluster_mask]

            if cluster_embeddings.shape[0] > 0:
                cluster_rate = rate_distortion_function(cluster_embeddings, delta)
                cluster_rates.append(cluster_rate)

        if len(cluster_rates) > 0:
            cluster_rates = torch.tensor(cluster_rates, device=embeddings.device)
            average_cluster_rate = torch.mean(cluster_rates)
        else:
            average_cluster_rate = torch.tensor(0.0, device=embeddings.device)

        return average_cluster_rate - overall_rate

    class Dataset(GeneralModel.Dataset):
        def __init__(self, model, corpus, phase):
            super().__init__(model, corpus, phase)

        def _get_feed_dict(self, index):
            user_id, target_item = self.data['user_id'][index], self.data['item_id'][index]
            if self.phase == 'train':  # and self.model.test_all:
                neg_items = np.arange(1, self.corpus.n_items)
            else:
                neg_items = self.data.get('neg_items', np.array([]))[index]
            item_ids = np.concatenate([[target_item], neg_items]).astype(int)
            feed_dict = {
                'user_id': user_id,
                'item_id': item_ids,
            }
            return feed_dict    
        
    # def kmeans_torch(self, X, n_clusters, max_iter=100, tol=0.001):
    #     n_samples, n_features = X.shape
    #     centers = X[torch.randperm(n_samples)[:n_clusters]]  # 随机初始化聚类中心
    #     for _ in range(max_iter):
    #         distances = torch.cdist(X, centers)
    #         labels = torch.argmin(distances, dim=1)
    #         new_centers = torch.stack([X[labels == i].mean(dim=0) for i in range(n_clusters)])
    #         if torch.norm(centers - new_centers) < tol:
    #             break
    #         centers = new_centers
    #     return labels