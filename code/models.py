'''
DRRNN
@author:
WuDong Xi (xiwd@mail3.sysu.edu.cn)
@ created:
18/8/2019
@references:
'''
# -*- coding: utf-8 -*-

import os
from time import time
import pickle
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from DRRNN import DRRNN
from evaluate import RMSE,MAE,evaluate_model

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

class subDataset(Dataset):
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label
    def __len__(self):
        return len(self.Data)
    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index])
        label = torch.Tensor(self.Label[index])
        return data, label

def weights_init(m):
    if isinstance(m,nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

 
def loss_function(predicted_ratings, true_ratings, predicted_embeddings, vae_embeddings, lambda_r): 
    review_loss = nn.MSELoss(size_average = False) 
    rating_loss = nn.MSELoss(size_average = False)
    RTS = rating_loss(predicted_ratings,true_ratings) 
    RES = review_loss(predicted_embeddings, vae_embeddings)
    loss = RTS + RES.mul(lambda_r)
    return loss


def train(input_data_path,rating_train, rating_test ,res_path,
    learning_rate, num_factors, batch_size, num_epochs, word2vector, dim,kernel_sizes,kernel_num,lambda_r,dropout):

    
    print "=================================Load input data==========================================="

    print "load the parameter"
    para_file = open(os.path.join(input_data_path, 'para'),'rb')
    para = pickle.load(para_file)
    user_num = para['user_num']
    item_num = para['item_num']
    u_review_num = para['u_review_num']
    i_review_num = para['i_review_num']
    u_review_len = para['u_review_len']
    i_review_len = para['i_review_len']
    t_review_num = para['t_review_num']
    t_review_len = para['t_review_len']
    vocabulary_user = para['user_vocab']
    vocabulary_item = para['item_vocab']
    vocabulary_review = para['review_vocab']
    train_length = para['train_length']
    u_text = para['u_text']
    i_text = para['i_text']
    review_embedding = para['review_embedding']
    weight_user = para['weight_user']
    weight_item = para['weight_item']
    para_file.close()

    
    print "load the train data"
    train_data_file = open(os.path.join(input_data_path, 'train'),'rb')
    y_train = pickle.load(train_data_file)
    train_data_file.close()
    
    print "============================Training the model DRRNN======================================"


    print "Load the train_matrix"
    train_matrix= get_train_matrix(rating_train, user_num, item_num)
    rating_matrix = torch.tensor(train_matrix, dtype = torch.float, device = device)
    train_data = subDataset(y_train, review_embedding)

    print "Load the valid/test data"
    test_data, test_labels =get_valid_and_test_data(rating_test)
    test_data = DataLoader(test_data,batch_size,shuffle = False, num_workers = 2)

    # convert the dictionary to the list
    u_text2 = np.zeros([user_num + 1, u_review_num * u_review_len], dtype = np.long)
    for i in u_text.keys():
        u_text2[i] = np.array(u_text[i]).reshape(-1)
    u_text = u_text2
    i_text2 = np.zeros([item_num + 1, i_review_num * i_review_len], dtype = np.long)
    for i in i_text.keys():
        i_text2[i] = np.array(i_text[i]).reshape(-1)
    i_text = i_text2
    u_text = torch.tensor(u_text, dtype = torch.long, device = device)
    i_text = torch.tensor(i_text, dtype = torch.long, device = device)
    vocabulary_user_size = len(vocabulary_user)
    vocabulary_item_size = len(vocabulary_item)

    # Build the DRRNN model
    model_out_file = os.path.join(res_path, 'DRRNN.pt')
    model = DRRNN(rating_matrix, user_num, item_num, num_factors, vocabulary_user_size, vocabulary_item_size, weight_user, weight_item,dim, kernel_sizes, kernel_num,
        dropout,u_text, i_text)
    model.to(device)


    start_time = time()
    res_file = open(os.path.join(res_path,'state.log'), 'w')
    rmse, mae = evaluate_model(model, test_data,test_labels)
    print "Init: rmse = %.4f, mae = %.4f [%.1f s]" % (rmse, mae, time() - start_time)
    res_file.write("Init: rmse = %.4f, mae = %.4f [%.1f s]\n" % (rmse, mae, time() - start_time))
    best_rmse, best_mae, best_iter = rmse, mae, -1
    
    # Define the loss optimizer
    #optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum=0.95, nesterov=True) #learning_rate, for other datasets
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)

    for epoch in range(num_epochs):
        start_time = time()
        # Muti-process(num_workers) read the training data and shuffle the order of the samples
        trainData = DataLoader(train_data,batch_size,shuffle = True, num_workers = 4) 
        model.train()
        train_loss = 0
        for i, data in enumerate(trainData):
            y_train, review_embedding = data
            inputs, labels = y_train[:, :2], y_train[:, 2]
            inputs, labels, review_embedding = inputs.to(device, dtype=torch.long), labels.to(device, dtype = torch.float), review_embedding.to(device,dtype = torch.float)

            # zero the parameters in each batch
            optimizer.zero_grad()
            predicted_ratings, predicted_embeddings = model(inputs) #
            loss = loss_function(predicted_ratings.squeeze(), labels, predicted_embeddings.squeeze(),review_embedding, lambda_r) #
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        #scheduler.step()

        train_time = time()
        model.eval()
        rmse, mae = evaluate_model(model, test_data, test_labels, True)
        print "Iteration %d [%.1f s]: rmse = %.4f, mae = %.4f , loss = %.4f [%.1f s]" \
        % (epoch, train_time - start_time, rmse, mae, train_loss, time() - start_time)
        res_file.write("Iteration %d [%.1f s]: rmse = %.4f, mae = %.4f , loss = %.4f [%.1f s] \n" \
        % (epoch, train_time - start_time, rmse, mae, train_loss, time() - start_time))
        if rmse < best_rmse:
            best_rmse, best_mae, best_iter = rmse, mae, epoch
            torch.save(model.state_dict(), model_out_file)

    print "Done. Best Iteration %d: rmse = %.4f, mae = %.4f." % (best_iter, best_rmse, best_mae)
    res_file.write("Done. Best Iteration %d: rmse = %.4f, mae = %.4f.\n" % (best_iter, best_rmse, best_mae))
    print "The best DRRNN model is saved to %s\n" % model_out_file
    res_file.write("The best DRRNN model is saved to %s" % model_out_file)



def get_train_matrix(rating_train,user_num, item_num):
    file_train = open(rating_train, "r")
    train_matrix = np.zeros([user_num + 1, item_num + 1], dtype = np.float)
    train_data = []
    for line in file_train:
        line = line.split(',')
        train_matrix[int(line[0]), int(line[1])] = line[2]
        train_data.append([long(line[0]), long(line[1]), float(line[2])])
    file_train.close()
    return train_matrix

def get_valid_and_test_data(rating_valid):

    file_valid = open(rating_valid, "r")
    valid_data = []
    valid_labels = []
    for line in file_valid:
        line = line.split(',')
        valid_data.append([long(line[0]), long(line[1])])
        valid_labels.append(float(line[2]))

    return np.array(valid_data), valid_labels

