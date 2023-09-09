'''
@author:
WuDong Xi (xiwd@mail2.sysu.edu.cn)
@ created:
25/8/2019
@references:
'''
import numpy as np
import torch

def RMSE(predict_rating, labels):
    mse = np.sum((np.array(predict_rating) - np.array(labels))**2) * 1.0 /len(labels)
    rmse = np.sqrt(mse)
    return rmse

def MAE(predict_rating, labels):
    return np.sum(abs(predict_rating - np.array(labels))) / len(labels)

def evaluate_model(model,valid_data, labels, test = False):

    predict_rating = []
    for data in valid_data:
        with torch.no_grad():
            data = data.long().cuda() 
            prediction_ratings, prediction_vectors = model(data)
            prediction = (prediction_ratings.squeeze()).data.cpu().numpy()
            for i in prediction:
                predict_rating.append(i)
    if test:
       predict_rating[predict_rating > 5] = 5.
       predict_rating[predict_rating < 1] = 1.
    rmse = RMSE(predict_rating,labels)
    mae = MAE(predict_rating,labels)
    return rmse,mae


