import pandas as pd
import numpy as np
from sklearn.metrics import log_loss

pred_files = ['prediction_xgboost.csv','prediction_inceptionV3.csv']
preds = ['../output/'+x for x in pred_files]

def computeMetrics(pred_path):
    # compute predict label and true label
    pred = pd.read_csv(pred_path)
    pred['pred_label'] = (pred['friedChicken']>0.5).astype(int)
    def map_to_true_label(r):
        ind = int(r['image'].split('_')[1].split('.')[0])
        return int(ind<=1000)
    pred['true_label'] = pred.apply(map_to_true_label,axis=1)

    # accuracy
    def get_accuracy():
        n_test = pred.shape[0]
        accuracy = 1 - np.sum(np.abs(pred['pred_label'] - pred['true_label']))*1.0/n_test
        return accuracy

    # logloss
    def get_logloss():
        return log_loss(pred['true_label'],pred[['labradoodle','friedChicken']])

    return [get_accuracy(),get_logloss()]

for pred in preds:
    model_name = pred.split('/')[-1].split('_')[1].split('.')[0]
    accuracy,logloss = computeMetrics(pred)
    print(model_name+' accuracy is:'+str(100*round(accuracy,4))+'%; '+'logloss is:'
    +str(round(logloss,4)))
