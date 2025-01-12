import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as BaseDataset
from torch.nn.utils.rnn import pad_sequence
from typing import List

from utils import utils
from models.BaseModel import BaseModel
#bpr 把用户和iteminput到一个向量空间里面做内积来评分
class BPR(BaseModel): 
	
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embeddings.')
		return BaseModel.parse_model_args(parser)


	def __init__(self, args, corpus):
		
		self.emb_size = args.emb_size
		self.user_num = corpus.n_users
		super().__init__(args, corpus)

	"""
	Key Methods
	"""
	def _define_params(self):
		# 定义模型参数
		self.i_embeddings = torch.nn.Embedding(self.item_num, self.emb_size)
		self.item_bias = torch.nn.Embedding(self.item_num, 1)
		self.u_embeddings = torch.nn.Embedding(self.user_num, self.emb_size)
		self.user_bias = torch.nn.Embedding(self.user_num,1)


	def forward(self, feed_dict: dict) -> dict:
		"""
		:param feed_dict: batch prepared in Dataset
		:return: out_dict, including prediction with shape [batch_size, n_candidates]
		"""
		u_ids = feed_dict['user_id']
		i_ids = feed_dict['item_id']  # [batch_size, n_candidates]
		u_vectors = self.u_embeddings(u_ids)
		i_vectors = self.i_embeddings(i_ids)
		u_bias = self.user_bias(u_ids)
		i_bias = self.item_bias(i_ids)
		predictions = torch.sum(u_vectors * i_vectors, dim=-1) + u_bias.squeeze() + i_bias.squeeze()
		return {'predictions': predictions}
	
	"""
	Define Dataset Class
	"""
	class Dataset(BaseModel.Dataset):

		# ! Key method to construct input data for a single instance
		def _get_feed_dict(self, index):
			feed_dict = super()._get_feed_dict(index)
			feed_dict['user_id'] = self.data['user_id'][index]
            return feed_dict
