'''
DRRNN
@author:
WuDong Xi (xiwd@mail2.sysu.edu.cn)
@ created:
15/8/2019
@references:
'''
import argparse
import sys
import os
import pickle
import numpy as np
import torch
import time

from load_data import data_factory
from data_pro import data_process
from models import train


parser = argparse.ArgumentParser()

# Option for pre-processing data
parser.add_argument("-c", "--do_preprocess", type=bool,
                    help="True or False to preprocess raw data for DRRNN (default = False)", default=False)
parser.add_argument("-r", "--raw_data_path", type=str,
                    help="Path to raw rating and review data.")
parser.add_argument("-d","--input_data_path", type=str,
                    help="Path to input training, valid, test data")
parser.add_argument("-p", "--word2vec", type = str,
                    help="Path to the word2vec file with pre-trained embeddings")
parser.add_argument("-t", "--split_ratio", type=float,
                    help="Ratio: 1-ratio, ratio/2 and ratio/2 of the entire dataset (R) will be training, valid and test set, respectively (default = 0.2)", default=0.2)



# Option for pre-processing data and running DRRNN
parser.add_argument("-rt", "--rating_data_path", type=str,
                    help="Path to the training, valid and test rating data sets")
parser.add_argument("-re", "--review_data_path", type=str, 
                    help="Path to the training, valid and test review data sets")

# Option for running DRRNN
parser.add_argument("-o", "--res_path", type=str,
                    help="Path to DRRNN's result")
parser.add_argument("-l", "--learning_rate", type = float,
                    help="the learning of the optimizer methods", default = 0.000005)
parser.add_argument("-f","--num_factors", type=int,
                    help = "the number of factors for the review embedding and user_item embedding",default=64)
parser.add_argument("-b","--batch_size", type = int,
                    help = "the number of the samples in one batch", default = 100)
parser.add_argument("-e","--num_epochs",type = int,
                    help = "the number of epochs in the DRRNN training process", default = 20)
parser.add_argument("-ne","--num_epochs_vae", type = int,
                    help = "the number of epochs in the DAE training process", default = 300)
parser.add_argument("-m","--dim", type = int,
                    help = "the dimension of the word embedding", default = 300)
parser.add_argument("-ks","--kernel_sizes", type = list,
                    help = "the size of the CNN kernel", default = [3,4,5])
parser.add_argument("-kn", "--kernel_num",type = int,
                    help = "the number of the CNN kernels", default = 100) # Patio 50
parser.add_argument("-a","--lambda_r", type = float,
                    help = "the hyper parameter used for balancing the rating loss and the review loss", default = 0.5)
parser.add_argument("-u","--dropout", type = float,
                    help = "the dropout ratio", default = 0.5)
parser.add_argument("-pt","--percent_of_text",type = float,
                    help = "the percent of the max len words in reviews", default = 0.5)
parser.add_argument("-pn","--percent_of_num",type = float,
                    help = "the percent of the max number of reviews for users and items", default = 0.5)

def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

init_seed()
args = parser.parse_args()
do_preprocess = args.do_preprocess 
rating_data_path = args.rating_data_path 
review_data_path = args.review_data_path
input_data_path = args.input_data_path
word2vec = args.word2vec
dim = args.dim
res_path = args.res_path
num_factors = args.num_factors
batch_size = args.batch_size

if rating_data_path is None:
    sys.exit("Argument missing - rating_data_path is required")
if review_data_path is None:
    sys.exit("Argument missing - review_data_path is required")
if input_data_path is None:
    sys.exit("Argument missing - input_data_path is required")
if res_path  is None:
    sys.exit("Argument missing - result_path is required")

if do_preprocess:
    raw_data_path = args.raw_data_path
    split_ratio = args.split_ratio
    num_epochs_vae = args.num_epochs_vae
    percent_of_text = args.percent_of_text
    percent_of_num = args.percent_of_num
    print "=================================Preprocess Option Setting================================="
    print "\tsaving preprocessed rating data path - %s" % rating_data_path
    print "\tsaving preprocessed review data path - %s" % review_data_path
    print "\tsplit_ratio: %.1f" % split_ratio
    print "==========================================================================================="



    DataFactory = data_factory(raw_data_path,rating_data_path,review_data_path,split_ratio)  # Initialize the data class
    data = DataFactory.read_data()
    DataFactory.get_rating_and_review_data(data)

    rating_train = os.path.join(rating_data_path, 'rating_train.csv')
    rating_test = os.path.join(rating_data_path, 'rating_test.csv')

    user_review = os.path.join(review_data_path, 'user_review')
    item_review = os.path.join(review_data_path, 'item_review')
    train_review = os.path.join(review_data_path,'train_reviews')
    data_process(rating_train, user_review, item_review, input_data_path,res_path,train_review, \
    dim,word2vec,num_epochs_vae,num_factors, percent_of_text,percent_of_num)

else:
    
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    kernel_sizes = args.kernel_sizes
    kernel_num = args.kernel_num
    lambda_r = args.lambda_r
    dropout = args.dropout

    print "The After: only one layer"

    print "===================================DRRNN Option Setting==================================="
    print "\trating data path - %s" % rating_data_path
    print "\treview data path - %s" % review_data_path
    print "\tresult path - %s" % res_path
    print "\tlearning_rate: %.5f\n\tnum_factors: %d\n\tbatch_size: %d\n\tnum_epochs: %d\n\tkernel_num: %d\n\tlambda_r: %.3f\n\tdropout: %.3f" \
        % (learning_rate, num_factors, batch_size, num_epochs,kernel_num,lambda_r,dropout)
    print "\tkernel_sizes: ", kernel_sizes
    print "==========================================================================================="
    rating_train = os.path.join(rating_data_path, 'rating_train.csv')
    rating_test = os.path.join(rating_data_path, 'rating_test.csv')
    train(input_data_path,rating_train, rating_test,res_path,learning_rate, num_factors, batch_size, num_epochs, word2vec, dim,kernel_sizes,kernel_num,lambda_r,dropout)


