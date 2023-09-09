'''
DRVNN
@author:
WuDong Xi (xiwd@mail2.sysu.edu.cn)
@ created:
21/8/2019
@references:
'''
import numpy as np
from time import time


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

class DRVNN(nn.Module):
	"""
	DRVNN: Only utilize the reviews to predict the target ratings
	input: user/item reviews for the corresponding ratings
	output: the predicted rating for the given user and item
	"""
	def __init__(self, train_rating_matrix, num_users, num_items, num_factors,vocabulary_user_size, vocabulary_item_size, weight_user, weight_item,vocab_embedding_size, kernel_sizes, kernel_num,
		dropout,user_reviews, item_reviews):
		super(DRVNN, self).__init__()
		self.train_rating_matrix = train_rating_matrix
		self.user_reviews = user_reviews
		self.item_reviews = item_reviews

		# CNN for user and item review text
		self.user_review_embedding = nn.Embedding(vocabulary_user_size, vocab_embedding_size)
		self.user_review_embedding.weight.data.copy_(torch.from_numpy(weight_user))
		self.user_convs1 = nn.ModuleList([nn.Conv2d(1,kernel_num,(kernel_size, vocab_embedding_size)) for kernel_size in kernel_sizes])
		"""
		# self.user_conv13 = nn.Conv2d(in_channels = 1, outchannels = kernel_num, 
		# kernel_size = (kernel_size, vocab_embedding_size))
        self.conv13 = nn.Conv2d(1, kernel_num, (3, vocab_embedding_size))
        self.conv14 = nn.Conv2d(1, kernel_num, (4, vocab_embedding_size))
        self.conv15 = nn.Conv2d(1, kernel_num, (5, vocab_embedding_size))		
		"""
		self.item_review_embedding = nn.Embedding(vocabulary_item_size, vocab_embedding_size)
		self.item_review_embedding.weight.data.copy_(torch.from_numpy(weight_item))
		self.item_convs1 = nn.ModuleList([nn.Conv2d(1,kernel_num,(kernel_size, vocab_embedding_size)) for kernel_size in kernel_sizes])


		self.dropout_user = nn.Dropout(dropout)
		self.dropout_item = nn.Dropout(dropout)
		self.Fusion_review = nn.Linear(kernel_num * len(kernel_sizes) * 2, num_factors)  

		# Fusion
		self.predict = nn.Linear(num_factors,1)

	def conv_and_pool(self, x, conv):
		x = F.relu(conv(x).squeeze(3)) # (N, kernel_num, vocab_embedding_size - kernel_size + 1)
		x = F.max_pool1d(x, x.size(2)).squeeze(2)
		return x

	def forward(self, x):
		
		# input_data
		user_index= x[:,0]
		item_index = x[:,1]
		user_review = self.user_reviews[user_index,:]
		item_review = self.item_reviews[item_index,:]
	
		# user and item review latent factor
		user_review_em = self.user_review_embedding(user_review)
		item_review_em = self.item_review_embedding(item_review)
		user_review_em = user_review_em.unsqueeze(1)
		item_review_em = item_review_em.unsqueeze(1)

		# [(N, kernel_num, vocab_embedding_size - kernel_size + 1), ...] * len(Ks) 
		user_review_cnn = [F.relu(conv(user_review_em)).squeeze(3) for conv in self.user_convs1]
		# [(N, kernel_num), ...] * len(Ks) 
		user_review_maxpool = [F.max_pool1d(i,i.size(2)).squeeze(2) for i in user_review_cnn]

		item_review_cnn = [F.relu(conv(item_review_em)).squeeze(3) for conv in self.item_convs1]
		item_review_maxpool = [F.max_pool1d(i,i.size(2)).squeeze(2) for i in item_review_cnn]

		user_review_cat = torch.cat(user_review_maxpool,1)
		item_review_cat = torch.cat(item_review_maxpool,1)

		# dropout
		user_review_cat = self.dropout_user(user_review_cat)
		item_review_cat = self.dropout_item(item_review_cat)
		
		review_predict_vector = torch.tanh(self.Fusion_review(torch.cat((user_review_cat, item_review_cat), 1)))

		out = self.predict(review_predict_vector)

		return out, review_predict_vector











		




		