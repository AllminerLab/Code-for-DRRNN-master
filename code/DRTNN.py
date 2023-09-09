'''
DRRNN
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

class DRTNN(nn.Module):
	"""
	DRTNN: Only utilize the ratings to predict the target ratings
	input: user/item rating vectors for the corresponding ratings
	output: the predicted rating for the given user and item
	"""
	def __init__(self, train_rating_matrix, num_users, num_items, num_factors,vocabulary_user_size, vocabulary_item_size, weight_user, weight_item,vocab_embedding_size, kernel_sizes, kernel_num,
		dropout,user_reviews, item_reviews):
		super(DRTNN, self).__init__()
		self.train_rating_matrix = train_rating_matrix

		# MLP for user and item rating vectors
		self.user_rating_embedding = nn.Linear(num_items + 1, 256, bias = False)
		self.item_rating_embedding = nn.Linear(num_users + 1, 256, bias = False)
		self.rating_layer1 = nn.Linear(512, 256)
		self.rating_layer2 = nn.Linear(256,128)
		self.rating_layer3 = nn.Linear(128,num_factors)

		# Fusion
		self.predict = nn.Linear(num_factors,1)

	def forward(self, x):
		
		# input_data
		user_index= x[:,0]
		item_index = x[:,1]
		
		user_vector = self.train_rating_matrix[user_index, :]
		item_vector = self.train_rating_matrix[:, item_index].t()

		# user and item rating vectors
		user_rating = self.user_rating_embedding(user_vector)
		item_rating = self.item_rating_embedding(item_vector)
		rating_hidden1 = F.relu(self.rating_layer1(torch.cat((user_rating,item_rating), 1)))
		rating_hidden2 = F.relu(self.rating_layer2(rating_hidden1))
		rating_predictive_vector = F.relu(self.rating_layer3(rating_hidden2))

		out = self.predict(rating_predictive_vector)

		return out











		




		