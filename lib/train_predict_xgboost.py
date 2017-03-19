# code heavily adapted from https://www.kaggle.com/sudalairajkumar/two-sigma-connect-rental-listing-inquiries/xgb-starter-in-python/notebook
import os
import sys
import operator
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=1000):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.1
    param['max_depth'] = 6
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X,label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test')]
        model = xgb.train(plist,xgtrain,num_rounds,watchlist,early_stopping_rounds=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plist, xgtrain, num_rounds)
    
    pred_test_y = model.predict(xgtest)
    return pred_test_y, model

input_path = '../../localData/prj3/training_data/sift_features/sift_features.csv'

total_df = pd.read_csv(input_path).transpose()

labels = [1 for i in range(1000)] + [0 for i in range(1000)]
total_df['label'] = labels

total_df = utils.shuffle(total_df)

train_test_ratio = 0.8
n_total = total_df.shape[0]
n_train = int(n_total*train_test_ratio)

train_df = total_df.iloc[:n_train]
test_df = total_df.iloc[n_train:]

train_X = train_df.ix[:,:5000]
train_y = train_df['label']

test_X = test_df.ix[:,:5000]

train_X = sparse.csr_matrix(train_X.values)
test_X = sparse.csr_matrix(test_X.values)

train_y = np.array(train_y)

#xgtest = xgb.DMatrix(test_X)
preds, _ = runXGB(train_X, train_y, test_X, num_rounds=400)
#preds = model.predict(xgtest, ntree_limit=model.best_ntree_limit)
out_df = pd.DataFrame(preds)
out_df.columns = ["labradoodle","friedChicken"]
out_df["image_id"] = test_df.index
out_df.to_csv("../output/prediction_xgboost.csv", index=False)