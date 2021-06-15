import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from collections import OrderedDict
import numpy as np

def get_performance(y_trues, y_preds):
    fpr, tpr, t = roc_curve(y_trues, y_preds)
    roc_score = auc(fpr, tpr)
    
    #Threshold
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(t, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    threshold = roc_t['threshold']
    threshold = list(threshold)[0]
    
    y_preds = [1 if ele >= threshold else 0 for ele in y_preds] 
    
    
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_trues, y_preds, average="binary", pos_label=0)
    #### conf_matrix = [["true_normal", "false_abnormal"], ["false_normal", "true_abnormal"]]     
    conf_matrix = confusion_matrix(y_trues, y_preds)
    performance = OrderedDict([ ('AUC', roc_score), ('precision', precision),
                                ("recall", recall), ("F1_Score", f1_score), ("conf_matrix", conf_matrix),
                                ("threshold", threshold)])
                                
    return performance