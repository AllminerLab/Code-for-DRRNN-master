import numpy as np
import os
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

class AE(nn.Module):
	"""
	VDE: Obtain the latent embedding for all the reviews
	input: user review, item review
	output: the latent embedding for all the reviews
	"""
	def __init__(self,input_size,hidden_layer_size, Vocabulary_size, embedding_size,latent_variable_size,pretrained_weight):
		super(AE, self).__init__()
		
		self.embedding = nn.Embedding(Vocabulary_size, embedding_size)
		self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
		self.embedding.weight.requires_grad = False
		self.encoder_layer1 = nn.Linear(input_size, 1024)
		self.encoder_layer2 = nn.Linear(1024,256)
		self.encoder_layer3 = nn.Linear(256, hidden_layer_size)
		self.decoder_layer1 = nn.Linear(hidden_layer_size, 256)
		self.decoder_layer2 = nn.Linear(256,1024)
		self.decoder_layer3 = nn.Linear(1024, input_size)

	def encode(self, x):

		embedding_vector1 = torch.relu(self.encoder_layer1(x))
		embedding_vector2 = torch.relu(self.encoder_layer2(embedding_vector1))
		embedding_vector3 = torch.tanh(self.encoder_layer3(embedding_vector2))
		return embedding_vector3

	def decode(self, z):
		recon_x1 = torch.relu(self.decoder_layer1(z))
		recon_x2 = torch.relu(self.decoder_layer2(recon_x1))
		recon_x3 = torch.tanh(self.decoder_layer3(recon_x2))
		return recon_x3

	def forward(self,x):

		x = self.embedding(x)
		x = x.view(x.size()[0],-1)
		z = self.encode(x)
		recon_x = self.decode(z)
		#print "recon_x.size():",recon_x.size()

		return z,recon_x,x



def ae_train(input_size,hidden_layer_size,  num_epochs, 
	batch_size, learning_rate, review_data, res_path,Vocabulary_size, embedding_size,pretrained_weight ,latent_variable_size = 20):
	
	model = AE(input_size,hidden_layer_size, Vocabulary_size, embedding_size,latent_variable_size,pretrained_weight)
	model.to(device)

	model_out_file = os.path.join(res_path, 'AE.pt')
    	# Define the loss optimizer
	criterion = nn.MSELoss(size_average = False)
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	for epoch in range(num_epochs):
	
		start_time = time()
		train_loss = 0
		data_loader = DataLoader(review_data,batch_size,shuffle = True, num_workers = 2)

		for batch_idx, data in enumerate(data_loader):
			optimizer.zero_grad()
			data = data.to(device, dtype=torch.long)
			embedding_vector, recon_data, embedding_data= model(data)
			loss = criterion(recon_data, embedding_data)
			loss.backward()
			train_loss += loss.item()
			optimizer.step()
		train_loss = train_loss / len(review_data)

		print "Iteration %d : Average loss = %.4f [%.1f s]" \
		% (epoch, train_loss, time() - start_time)

	review_data = DataLoader(review_data, batch_size, shuffle = False, num_workers = 2)
	embedding = []
	for i, data in enumerate(review_data):
		
		with torch.no_grad():
			data = data.to(device, dtype = torch.long)
			embedding_vectors, recon_data, embedding_data = model(data)
			for embedding_vector in (embedding_vectors.squeeze()).cpu().numpy():
				embedding.append(embedding_vector)
	torch.save(model.state_dict(), model_out_file)

	return embedding

