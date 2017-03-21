# code heavily adapted from https://www.kaggle.com/sudalairajkumar/two-sigma-connect-rental-listing-inquiries/xgb-starter-in-python/notebook
import os
from time import time 
import sys
import operator
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss

input_path = '../data/sift_features.csv'

def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=1000):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.1
    param['max_depth'] = 6
    param['silent'] = 1
    param['num_class'] = 2
    param['eval_metric'] = "logloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    num_rounds = num_rounds

    plist = list(param.items())
    xgtrain = xgb.DMatrix(train_X,label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test')]
        train_start = time()
        model = xgb.train(plist,xgtrain,num_rounds,watchlist,early_stopping_rounds=20)
        train_end = time()
    else:
        xgtest = xgb.DMatrix(test_X)
        train_start = time()
        model = xgb.train(plist, xgtrain, num_rounds)
        train_end = time()
    
    test_start = time()
    pred_test_y = model.predict(xgtest)
    test_end = time()
    return pred_test_y, model, (train_end - train_start), (test_end - test_start)

#input_path = '../../localData/prj3/training_data/sift_features/sift_features.csv'

fix_test_path = '../output/prediction_inceptionV3.csv'
test_images = pd.read_csv(fix_test_path)['image']
test_images = [x.split('.')[0] for x in test_images.tolist()]

total_df = pd.read_csv(input_path).transpose()

labels = [0 for i in range(1000)] + [1 for i in range(1000)]
total_df['label'] = labels

# for ensemble purpose make sure test set the same as in inceptionV3 model
total_df.index = [x.split('.')[0] for x in total_df.index.tolist()]
test_df = total_df.ix[total_df.index.isin(test_images)]
train_df = total_df.ix[~total_df.index.isin(test_images)]
'''
total_df = utils.shuffle(total_df)

train_test_ratio = 0.8
n_total = total_df.shape[0]
n_train = int(n_total*train_test_ratio)

train_df = total_df.iloc[:n_train]
test_df = total_df.iloc[n_train:]
'''
k = total_df.shape[1] - 1
train_X = train_df.ix[:,:k]
train_y = train_df['label']

test_X = test_df.ix[:,:k]

train_X = sparse.csr_matrix(train_X.values)
test_X = sparse.csr_matrix(test_X.values)

train_y = np.array(train_y)

#xgtest = xgb.DMatrix(test_X)
preds, _ , training_time, predicting_time = runXGB(train_X, train_y, test_X, 
num_rounds=400)
#preds = model.predict(xgtest, ntree_limit=model.best_ntree_limit)
print('training time is:'+str(round(training_time,2))+'seconds;\npredicting time is:'+
str(round(predicting_time,2))+'seconds.')
out_df = pd.DataFrame(preds)
out_df.columns = ["friedChicken","labradoodle"]
out_df["image"] = test_df.index
out_df.to_csv("../output/prediction_xgboost_sift.csv", index=False)