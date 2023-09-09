import numpy as np
import re
import itertools
from collections import Counter

import tensorflow as tf
import csv
import pickle
import os
import sys

from AE import AE,ae_train

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # re.sub(r"A","a",s) replace 'A'  with 'a'
    string = re.sub(r"[^A-Za-z]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def read_pretrained_word2vec(word2vector, vocab, dim):

    if os.path.isfile(word2vector):
            raw_word2vec = open(word2vector, 'r')
    else:
            print "Path (word2vec) is wrong!"
            sys.exit()
    #print t_text
    word2vec_dic = {}
    all_line = raw_word2vec.read().splitlines()
    mean = np.zeros(dim)
    count = 0
    for line in all_line:
        tmp = line.split()
        _word = tmp[0]
        _vec = np.array(tmp[1:], dtype=float)
        word2vec_dic[_word] = _vec
        if _vec.shape[0] != dim:
                print "Mismatch the dimension of pre-trained word vector with word embedding dimension!"
                sys.exit()
        mean = mean + _vec
        count = count + 1

    mean = mean / count

    weight = np.random.uniform(-1.0, 1.0, (len(vocab), dim))
    count = 0
    for word in vocab:
        if word2vec_dic.has_key(word):
            weight[vocab[word]] = word2vec_dic[word]
            count = count + 1
        else:
            weight[vocab[word]] = np.random.normal(mean, 0.1, size=dim)
    print "There are %d words in the pre-trained word2vector" % count
    return weight

def pad_sentences(u_text, u_review_num, u_review_len, padding_word="<PAD/>"):
    """
    Pads all user's reviews to the same number.
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    review_num = u_review_num
    review_len = u_review_len

    u_text2 = {}
    for i in u_text.keys():
        u_reviews = u_text[i]
        padded_u_train = []
        for ri in range(review_num):
            if ri < len(u_reviews):
                sentence = u_reviews[ri]
                if review_len > len(sentence):
                    num_padding = review_len - len(sentence)
                    new_sentence = sentence + [padding_word] * num_padding
                    padded_u_train.append(new_sentence)
                else:
                    new_sentence = sentence[:review_len]
                    padded_u_train.append(new_sentence)
            else:
                new_sentence = [padding_word] * review_len
                padded_u_train.append(new_sentence)
        u_text2[i] = padded_u_train

    return u_text2

def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    # Counter: return {entry : count}
    # itertools.charin: contact the entries of the sequence
    # *: Mutiple non-critical parameter input
    word_counts1 = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    # most_common(): return the top-n list
    vocabulary_inv1 = [x[0] for x in word_counts1.most_common()]
    vocabulary_inv1 = list(sorted(vocabulary_inv1))
    # Mapping from word to index
    vocabulary1 = {x: i for i, x in enumerate(vocabulary_inv1)}

    return [vocabulary1, vocabulary_inv1]



def build_input_data(text, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    l = len(text)
    text2 = {}
    for i in text.keys():
        reviews = text[i]
        t = np.array([[vocabulary[str(word)] for word in words] for words in reviews])
        text2[i] = t
    
    return text2


def load_data_and_labels(rating_train, user_review, item_review, train_review,percent_of_text,percent_of_num):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files

    print "load the trainning data"
    file_train = open(rating_train, "r")
    file_user_review = open(user_review)
    file_item_review = open(item_review)
    file_train_review = open(train_review)

    user_reviews = pickle.load(file_user_review)
    item_reviews = pickle.load(file_item_review)
    train_reviews = pickle.load(file_train_review)


    file_user_review.close()
    file_item_review.close()
    file_train_review.close()


    y_train = []    # store the rating according to the order of the training data
    u_text = {}     # the all reviews of user
    i_text = {}     # the all reviews of item
    t_text = {}     # keep the same form of user review text and item review text
    t_text[0] = []
    i = 0

    for s in train_reviews:
        s1 = clean_str(s)
        s1 = s1.split(" ")
        t_text[0].append(s1)

    for line in file_train:
        i = i + 1
        line = line.split(',')
        if not u_text.has_key(int(line[0])):
            u_text[int(line[0])] = [] # initiate the id user's review text
            for s in user_reviews[int(line[0])]:
                s1 = clean_str(s)
                s1 = s1.split(" ")
                u_text[int(line[0])].append(s1)

        if not i_text.has_key(int(line[1])):
            i_text[int(line[1])] = []
            for s in item_reviews[int(line[1])]:
                s1 = clean_str(s)
                s1 = s1.split(" ")
                i_text[int(line[1])].append(s1)
        y_train.append([int(line[0]),int(line[1]),float(line[2])])

    file_train.close()


    print "============================the length of each list====================="

    #itervalues: Get the values of the list and save the memory 
    review_num_u = np.array([len(x) for x in u_text.itervalues()]) 
    x = np.sort(review_num_u)
    u_review_num = x[int(percent_of_num * len(review_num_u)) - 1]  # the max number of the user's review
    #print  x[int(0.9 * len(review_num_u)):len(review_num_u) - 1]
    u_review_len = np.array([len(j) for i in u_text.itervalues() for j in i])
    x2 = np.sort(u_review_len)
    u_review_len = x2[int(percent_of_text * len(u_review_len)) - 1] # the max length of the user's review

    

    review_num_i = np.array([len(x) for x in i_text.itervalues()])
    y = np.sort(review_num_i)
    i_review_num = y[int(percent_of_num * len(review_num_i)) - 1]
    i_review_len = np.array([len(j) for i in i_text.itervalues() for j in i])
    y2 = np.sort(i_review_len)
    i_review_len = y2[int(percent_of_text * len(i_review_len)) - 1]

    t_review_num = len(t_text[0])
    t_review_len = np.array([len(i) for x in t_text.itervalues() for i in x])
    t2 = np.sort(t_review_len)
    t_review_len = t2[int(percent_of_text * len(t_review_len)) - 1]
    print "u_review_num:", u_review_num
    print "u_review_len:", u_review_len
    print "i_review_num:", i_review_num
    print "i_review_len:", i_review_len
    print "t_review_num:", t_review_num
    print "t_review_len:", t_review_len
    user_num = len(u_text)
    item_num = len(i_text)
    print "user_num:", user_num
    print "item_num:", item_num
    return [u_text, i_text,t_text,y_train, u_review_num, i_review_num,t_review_num, u_review_len, i_review_len,t_review_len,user_num,item_num]

def load_data(rating_train, user_review, item_review,train_review, dim,percent_of_text,percent_of_num):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    u_text, i_text, t_text,y_train, u_review_num, i_review_num,t_review_num,u_review_len, i_review_len,t_review_len, user_num, item_num = \
        load_data_and_labels(rating_train, user_review, item_review,train_review,percent_of_text,percent_of_num)
    print "load data done"

    u_text = pad_sentences(u_text, u_review_num, u_review_len)
    print "pad user done"
    i_text = pad_sentences(i_text, i_review_num, i_review_len)
    print "pad item done"
    t_text= pad_sentences(t_text,t_review_num, t_review_len)
    print "pad train reviews done"


    # without the pre-trained embeddings 
    user_voc = [xx for x in u_text.itervalues() for xx in x]
    item_voc = [xx for x in i_text.itervalues() for xx in x]
    review_voc = [xx for x in t_text.itervalues() for xx in x]

    vocabulary_user, vocabulary_inv_user = build_vocab(user_voc)
    u_text = build_input_data(u_text,vocabulary_user)
    print "build the input_data of the u_text done!"
    vocabulary_item, vocabulary_inv_item = build_vocab(item_voc)
    i_text = build_input_data(i_text, vocabulary_item)
    print "build the input_data of the i_text done!"
    vocabulary_review, vocabulary_inv_review = build_vocab(review_voc)
    t_text = build_input_data(t_text,vocabulary_review)
    print "build the input_data of t_text done!"

    y_train = np.array(y_train)
    
    return [u_text, i_text,t_text, y_train, vocabulary_user, vocabulary_inv_user, vocabulary_item,
            vocabulary_inv_item,vocabulary_review,vocabulary_inv_review, user_num, item_num]

def data_process(rating_train, user_review, item_review, input_data_path, \
    res_path, train_review,dim,word2vector,num_epochs_ae,num_factors,percent_of_text,percent_of_num):

    u_text, i_text,t_text, y_train, vocabulary_user, vocabulary_inv_user, vocabulary_item, vocabulary_inv_item, \
    vocabulary_review,vocabulary_inv_review, user_num, item_num = \
        load_data(rating_train, user_review, item_review,train_review,dim,percent_of_text,percent_of_num)

    train_reviews = t_text[0]
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))

    
    y_train = y_train[shuffle_indices]
    train_reviews = train_reviews[shuffle_indices]

    batches_train = y_train
    t_text[0] = train_reviews

    print '==================================Begin to write the input data==========================='
    output = open(os.path.join(input_data_path, 'train'), 'wb')
    pickle.dump(batches_train, output)
    output.close()

    
    embedding_weight = read_pretrained_word2vec(word2vector,vocabulary_review,dim)
    weight_user = read_pretrained_word2vec(word2vector, vocabulary_user, dim)
    weight_item = read_pretrained_word2vec(word2vector, vocabulary_item,dim)

    t_review_len = t_text[0].shape[1]
    input_size = dim * t_review_len
    learning_rate = 0.0002
    batch_size = 256
    review_embedding = ae_train(input_size,num_factors,num_epochs_ae, \
        batch_size, learning_rate, train_reviews, res_path,len(vocabulary_review), dim, embedding_weight, t_review_len)
    para = {}
    para['user_num'] = user_num
    para['item_num'] = item_num
    para['u_review_num'] = u_text[0].shape[0]
    para['i_review_num'] = i_text[0].shape[0]
    para['t_review_num'] = t_text[0].shape[0]
    para['u_review_len'] = u_text[1].shape[1]
    para['i_review_len'] = i_text[1].shape[1]
    para['t_review_len'] = t_text[0].shape[1]
    para['user_vocab'] = vocabulary_user
    para['item_vocab'] = vocabulary_item
    para['review_vocab'] = vocabulary_review
    para['train_length'] = len(y_train)
    para['u_text'] = u_text
    para['i_text'] = i_text
    para['weight_user'] = weight_user
    para['weight_item'] = weight_item
    para['review_embedding'] = review_embedding
    output = open(os.path.join(input_data_path, 'para'), 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(para, output)
    output.close()

    

   










