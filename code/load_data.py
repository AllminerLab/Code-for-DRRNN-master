import os
import json
import pandas as pd
import pickle
import numpy as np


class data_factory(object):
    def __init__(self, raw_data_path,rating_data_path,review_data_path,split_ratio):
        self.raw_data_path = raw_data_path
        self.split_ratio = split_ratio
        self.rating_data_path = rating_data_path
        self.review_data_path = review_data_path
        
    def read_data(self):
        f= open(self.raw_data_path)
        users_id=[]
        items_id=[]
        ratings=[]
        reviews=[]

        for line in f:
            js=json.loads(line)
            if str(js['reviewerID'])=='unknown':
                print "unknown"
                continue
            if str(js['asin'])=='unknown':
                print "unknown2"
                continue
            reviews.append(js['reviewText'])
            users_id.append(str(js['reviewerID'])+',')
            items_id.append(str(js['asin'])+',')
            ratings.append(str(js['overall']))
        f.close()
        # Series: obtain the list[index user_id]
        data=pd.DataFrame({'user_id':pd.Series(users_id),
                           'item_id':pd.Series(items_id),
                           'ratings':pd.Series(ratings),
                           'reviews':pd.Series(reviews)})[['user_id','item_id','ratings','reviews']]

        users_id_count, items_id_count = self.get_count(data, 'user_id'), self.get_count(data, 'item_id')
        unique_user_id = users_id_count.index
        unique_item_id = items_id_count.index
        user2id = dict((uid, i) for (i, uid) in enumerate(unique_user_id))
        item2id = dict((sid, i) for (i, sid) in enumerate(unique_item_id))

        data = self.numerize(data, user2id, item2id)


        return data

    def get_count(self, data, id):
        # group by the corresponding column of id, and as_index = False indicates that
        # the index entry is displayed. 
        # return the list [id count]
        playcount_groupbyid = data[[id, 'ratings']].groupby(id, as_index=False) 
        count = playcount_groupbyid.size()
        return count


    def numerize(self, data, user2id, item2id):
        user_id = map(lambda x: user2id[x], data['user_id'])
        item_id = map(lambda x: item2id[x], data['item_id'])
        data['user_id'] = user_id
        data['item_id'] = item_id
        return data

    def get_rating_and_review_data(self, data):

        # Get the rating data
        ratings_data=data[['user_id','item_id','ratings']]
        print "the number of ratings: ", len(ratings_data)
        ratings_data.to_csv(os.path.join(self.rating_data_path, 'rating_data.csv'), index = False, header = None)
        num_ratings = ratings_data.shape[0]
        test = np.random.choice(num_ratings, size=int((self.split_ratio / 2) * num_ratings), replace=False)
        test_idx = np.zeros(num_ratings, dtype=bool)
        test_idx[test] = True

        # Split dataset into training set and test set (and valid set)
        rating_test = ratings_data[test_idx]
        rating_train = ratings_data[~test_idx]

        data_test =data[test_idx]
        data_train =data[~test_idx]

        # Split test_and_valid dataset into test set and valid set
        num_ratings_test = rating_test.shape[0]
        # test = np.random.choice(num_ratings_test_and_valid, size=int(0.50 * num_ratings_test_and_valid), replace=False)

        # test_idx = np.zeros(num_ratings_test_and_valid, dtype=bool)
        # test_idx[test] = True

        # rating_test = rating_test_and_valid[test_idx]
        # rating_valid = rating_test_and_valid[~test_idx]

        print "The length of Training ratings is {}".format(len(rating_train))
        print "The length of testing ratings is {}".format(len(rating_test))
        rating_train.to_csv(os.path.join(self.rating_data_path, 'rating_train.csv'), index=False,header=None)
        # rating_valid.to_csv(os.path.join(self.rating_data_path, 'rating_valid.csv'), index=False,header=None)
        rating_test.to_csv(os.path.join(self.rating_data_path, 'rating_test.csv'), index=False,header=None)

        print "Store the rating training set and test set in rating_train.csv and rating_test.csv separately"


        # Get the review data
        user_reviews={}    #user_id:  the all review
        item_reviews={}    #item_id:  the all review
        train_reviews = data_train['reviews'] #review: the review of the training data 
        for i in data_train.values:
            if user_reviews.has_key(i[0]):
                user_reviews[i[0]].append(i[3])
            else:
                user_reviews[i[0]]=[i[3]]
            if item_reviews.has_key(i[1]):
                item_reviews[i[1]].append(i[3])
            else:
                item_reviews[i[1]] = [i[3]]
        # for these users or items, who have none reviews
        # Set its reviews to '0'  
        for i in data_test.values:
            if not user_reviews.has_key(i[0]):
                user_reviews[i[0]]=['0']
            if not item_reviews.has_key(i[1]):
                item_reviews[i[1]] = ['0']

        pickle.dump(user_reviews, open(os.path.join(self.review_data_path, 'user_review'), 'wb'))
        pickle.dump(item_reviews, open(os.path.join(self.review_data_path, 'item_review'), 'wb'))
        pickle.dump(train_reviews, open(os.path.join(self.review_data_path,'train_reviews'), 'wb'))

        print "Store the user(item) reviews and rating_id in user(item)_review and user(item)_rid"
        print "Store the training reviews in the train_reviews"

















