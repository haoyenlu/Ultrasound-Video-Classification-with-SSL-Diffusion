import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm , t

import scipy
import pandas as pd
import seaborn as sn

def cal_CI(data, confidence = 0.95): 
    mean = np.mean(data)
    sem = scipy.stats.sem(data)
    ci = sem * scipy.stats.t.ppf((1 + confidence) / 2., len(data)-1) 
    return mean , ci

def cal_STD(data):
    mean = np.mean(data)
    std = np.std(data)
    return mean, std
    
def plot_average_roc_curve_of_all_models(model_metric_list, model_name_list):
    plt.rcParams.update({'font.size': 14})
    plt.rcParams["font.family"] = "Times New Roman"
    mean_fpr = np.linspace(0,1,100)

    for metric , name in zip(model_metric_list, model_name_list):
        print(f"==> {name}")
        tprs = []
        aucs = []

        for i in range(len(metric)): # 10 runs
            tprs_fold = []
            aucs_fold = []
            for j in range(4): # 4-folds
                fpr, tpr , _ = metric[i][j].cal_roc_curve()
                auc = metric[i][j].cal_auc_score()
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs_fold.append(interp_tpr)
                aucs_fold.append(auc)
            tprs.append(tprs_fold)
            aucs.append(aucs_fold)
        tprs = np.array(tprs)
        aucs = np.array(aucs)
    
        for i in range(4):
            mean , ci = cal_CI(aucs[:,i])
            print(f"Fold-{i+1} AUC:{mean:.3f} ({mean-ci:.3f}-{mean+ci:.3f})")
    
        average_tpr = []
        for j in range(4):
            tprs_fold =  tprs[:,j,:]
    
            mean_tpr, se_tpr = np.mean(tprs_fold,axis=0), scipy.stats.sem(tprs_fold,axis=0)
            average_tpr.append(mean_tpr)
    
        average_tpr = np.array(average_tpr)
        mean_tpr = average_tpr.mean(axis=0)
        mean, ci = cal_CI(aucs.flatten())
        plt.plot(mean_fpr,mean_tpr, linewidth=1.5, label=f'{name} (AUC={mean:.3f})')
        
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5, label='chance')
    plt.legend()
    plt.title(f"ROC Curves")
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.savefig("all_model_roc.png")
    plt.show()
        

def plot_roc_curves(all_metric , model_name):
    plt.rcParams.update({'font.size': 12})
    plt.rcParams["font.family"] = "Times New Roman"

    mean_fpr = np.linspace(0,1,100)
    tprs = []
    aucs = []
    
    for i in range(len(all_metric)):
        tprs_fold = []
        aucs_fold = []
        for j in range(4):
            fpr, tpr , _ = all_metric[i][j].cal_roc_curve()
            auc = all_metric[i][j].cal_auc_score()
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs_fold.append(interp_tpr)
            aucs_fold.append(auc)
        tprs.append(tprs_fold)
        aucs.append(aucs_fold)

        

    tprs = np.array(tprs)
    aucs = np.array(aucs)

    for i in range(4):
        mean , ci = cal_CI(aucs[:,i])
        print(f"Fold-{i+1} AUC:{mean:.3f}")

    average_tpr = []
    N , folds , _ = tprs.shape
    figure = plt.figure(figsize=(6,6))
    for j in range(folds):
        tprs_fold =  tprs[:,j,:]

        mean_tpr, se_tpr = np.mean(tprs_fold,axis=0), scipy.stats.sem(tprs_fold,axis=0)
        
        ci = se_tpr * scipy.stats.t.ppf((1 + 0.95) / 2., N-1) # 0.95 confidence interval
        
        plt.plot(mean_fpr, mean_tpr,color='b',label=f"fold{j+1} Mean ROC (AUC = {np.mean(aucs[:,j]):.3f})", alpha = 0.5)
        plt.fill_between(mean_fpr, mean_tpr - ci, mean_tpr + ci, color='blue', alpha=0.1)

        average_tpr.append(mean_tpr)

    average_tpr = np.array(average_tpr)
    mean_tpr = average_tpr.mean(axis=0)


    mean, ci = cal_CI(aucs.flatten())
    
    plt.plot(mean_fpr,mean_tpr, color="black",label="Mean ROC" , linewidth=2.5)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5, label='chance')
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.legend()
    plt.title(f"{model_name} Cross Validation (AUC = {mean:.3f}) ")
    return figure


import os
    
def save_result_object(result, result_dir):
    save_path = result_dir
    
    for i in range(len(result)): # 10 runs CV
        for j in range(len(result[i])): # 4-fold CV
            os.makedirs(os.path.join(save_path, f'cv-{i+1}'), exist_ok=True)
            save_object = {}
            save_object["Prediction"] = result[i][j].preds
            save_object["True"] = result[i][j].trues
            save_object["Labels"] = result[i][j].labels
            save_object["Info"] = result[i][j].info
            
            np.save(os.path.join(save_path, f'cv-{i+1}', f'fold-{j+1}.npy'), save_object) 


def load_result_object(result_dir, cv_repeat=10, num_fold = 4):
    from evaluate import Metric

    save_path = result_dir
    metric_list = []
    for i in range(cv_repeat): # 10 runs CV
        cv_list=  [] 
        for j in range(num_fold): # 4-fold
            data = np.load(os.path.join(save_path, f'cv-{i+1}', f'fold-{j+1}.npy'),allow_pickle=True).item()
            metric = Metric()
            metric.preds = data['Prediction']
            metric.trues = data['True']
            metric.labels = data['Labels']
            metric.info = data['Info']

            cv_list.append(metric)

        metric_list.append(cv_list)
    return metric_list