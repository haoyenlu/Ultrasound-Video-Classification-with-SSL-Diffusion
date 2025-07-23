from sklearn.metrics import roc_curve,roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, matthews_corrcoef

import numpy as np
import matplotlib.pyplot as plt

class Metric:
    def __init__(self, labels = [0,1] , info = ["patient_id","intervention"]):
        self.preds = []
        self.trues = []
        self.labels = labels
        self.target_names = ["Healthy","Unhealthy"]
        self.info = {label:[] for label in info}
    
    def update(self, y_true, y_pred):
        self.trues.extend(y_true)
        self.preds.extend(y_pred)
    
    def update_info(self, curr_info):
        for key, values in curr_info.items():
            self.info[key].extend(values)



    def reset(self):
        self.trues = []
        self.preds = []

    def cal_auc_score(self):
        auc = roc_auc_score(np.array(self.trues), np.array(self.preds), labels=self.labels)
        return auc

    def cal_roc_curve(self):
        fpr, tpr, thresholds = roc_curve(np.array(self.trues), np.array(self.preds), pos_label=1)
        return fpr, tpr , thresholds
    

    def cal_pr_curve(self):
        precision , recall , threshold = precision_recall_curve(np.array(self.trues), np.array(self.preds), pos_label=1)
        return precision ,recall , threshold

    def get_youden_index(self):
        fpr, tpr, thresholds = roc_curve(np.array(self.trues), np.array(self.preds), pos_label=1)
        youden = np.max(+tpr-fpr)
        youden_index = np.argmax(+tpr-fpr)
        optimal_thresholds = thresholds[youden_index]
        return youden, optimal_thresholds
    

    def get_optimal_threshold(self):
        fpr, tpr, thresholds = roc_curve(np.array(self.trues), np.array(self.preds), pos_label=1)
        l = {'accuracy':[],'mcc':[],'f1_score':[]}
        for t in thresholds:
            y_pred = (np.array(self.preds) >= (t - 1e-4)).astype(int)
            y_true = np.array(self.trues).astype(int)
            l['f1_score'].append(f1_score(y_true,y_pred))
            l['accuracy'].append(accuracy_score(y_true,y_pred))
            l['mcc'].append(matthews_corrcoef(y_true,y_pred))
        

        optimal_t = {'accuracy':thresholds[np.argmax(l['accuracy'])],'mcc':thresholds[np.argmax(l['mcc'])], 'f1_score':thresholds[np.argmax(l['f1_score'])]}
        return optimal_t
    
    def cal_metrics(self, return_threshold = False):
        fpr, tpr, thresholds = roc_curve(np.array(self.trues), np.array(self.preds), pos_label=1)
        youden_index = np.max(+tpr-fpr)
        
        metrics_threshold = {'accuracy':[], 'mcc':[],'f1_score':[] , 'thresholds': thresholds}
        for t in thresholds:
            y_pred = (np.array(self.preds) >= (t - 1e-4)).astype(int)
            y_true = np.array(self.trues).astype(int)

            metrics_threshold["accuracy"].append(accuracy_score(y_true,y_pred))
            metrics_threshold['mcc'].append(matthews_corrcoef(y_true,y_pred))
            metrics_threshold['f1_score'].append(f1_score(y_true,y_pred))

        optimal_f1_index = np.argmax(metrics_threshold['f1_score'])
        optimal_f1_threshold = thresholds[optimal_f1_index]

        y_pred = (np.array(self.preds) >= (optimal_f1_threshold - 1e-4)).astype(int).flatten()
        y_true = np.array(self.trues).astype(int).flatten()

        metrics_optimal = {
            'accuracy': np.max(metrics_threshold['accuracy']),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred),
            'mcc': np.max(metrics_threshold['mcc']),
            'f1_score': np.max(metrics_threshold['f1_score']),
            'youden_index': youden_index
        }

        if return_threshold: 
            return metrics_optimal, metrics_threshold
        
        return metrics_optimal 

       

