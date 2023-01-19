import torch
import numpy as np
from numpy.random import beta
from sklearn.metrics import average_precision_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model import  NetSmall as Net

from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, confusion_matrix

device=  torch.device("cuda" if torch.cuda.is_available() else "cpu")
from fairlearn.metrics import MetricFrame, demographic_parity_difference


def sample_batch_sen_idx(X, A, y, batch_size, s):    
    try :
        batch_idx = np.random.choice(np.where(A==s)[0], size=batch_size, replace=False).tolist()
    except:
        batch_idx = np.random.choice(np.where(A==s)[0], size=batch_size, replace=True).tolist()
    batch_x = X[batch_idx]
    batch_y = y[batch_idx]
    batch_x = torch.tensor(batch_x).to(device).float()
    batch_y = torch.tensor(batch_y).to(device).float()

    return batch_x, batch_y

def sample_batch_sen_idx_y(X, A, y, batch_size, s):
    batch_idx = []
    for i in range(2):
        idx = list(set(np.where(A==s)[0]) & set(np.where(y==i)[0]))
#         print("s:",s, "y:", i, "->", len(idx))
        try :
            batch_idx += np.random.choice(idx, size=batch_size, replace=False).tolist()
        except:
            batch_idx += np.random.choice(idx, size=batch_size, replace=True).tolist()

    batch_x = X[batch_idx]
    batch_y = y[batch_idx]
    batch_x = torch.tensor(batch_x).to(device).float()
    batch_y = torch.tensor(batch_y).to(device).float()

    return batch_x, batch_y

def sample_batch(X, A, y, batch_size):  
    batch_size = 4*batch_size
    batch_idx = np.random.choice(len(X), size=batch_size, replace=False).tolist()
    batch_x = X[batch_idx]
    batch_y = y[batch_idx]
    batch_a = A[batch_idx]
    
    idx_00 = list(set(np.where(batch_a==0)[0]) & set(np.where(batch_y==0)[0]))
    idx_01 = list(set(np.where(batch_a==0)[0]) & set(np.where(batch_y==1)[0]))
    idx_10 = list(set(np.where(batch_a==1)[0]) & set(np.where(batch_y==0)[0]))
    idx_11 = list(set(np.where(batch_a==1)[0]) & set(np.where(batch_y==1)[0]))
    
    batch_x = torch.tensor(batch_x).to(device).float()
    batch_y = torch.tensor(batch_y).to(device).float()
    batch_a = torch.tensor(batch_a).to(device).float()
    
    return batch_x, batch_y, batch_a, idx_00, idx_01, idx_10, idx_11 






def evaluate_eo(model, X_test, y_test, A_test):

    model.eval()
    idx_00 = list(set(np.where(A_test==0)[0]) & set(np.where(y_test==0)[0]))
    idx_01 = list(set(np.where(A_test==0)[0]) & set(np.where(y_test==1)[0]))
    idx_10 = list(set(np.where(A_test==1)[0]) & set(np.where(y_test==0)[0]))
    idx_11 = list(set(np.where(A_test==1)[0]) & set(np.where(y_test==1)[0]))

    X_test_00 = X_test[idx_00]
    X_test_01 = X_test[idx_01]
    X_test_10 = X_test[idx_10]
    X_test_11 = X_test[idx_11]

    y_test_00 = y_test[idx_00]
    y_test_01 = y_test[idx_01]
    y_test_10 = y_test[idx_10]
    y_test_11 = y_test[idx_11]

    X_test_00 = torch.tensor(X_test_00).to(device).float()
    X_test_01 = torch.tensor(X_test_01).to(device).float()
    X_test_10 = torch.tensor(X_test_10).to(device).float()
    X_test_11 = torch.tensor(X_test_11).to(device).float()

    pred_00 = model(X_test_00)
    pred_01 = model(X_test_01)
    pred_10 = model(X_test_10)
    pred_11 = model(X_test_11)

    gap_0 = pred_00.mean() - pred_10.mean()
    gap_1 = pred_01.mean() - pred_11.mean()
    gap_0 = abs(gap_0.data.cpu().numpy())
    gap_1 = abs(gap_1.data.cpu().numpy())

    gap_logit = gap_0 + gap_1

    # calculate average precision
    X_test_cuda = torch.tensor(X_test).to(device).float()
    output = model(X_test_cuda)
    y_scores = output[:, 0].data.cpu().numpy()
    ap = average_precision_score(y_test, y_scores)
    
    print(pred_00.mean().item(), pred_01.mean().item(), pred_10.mean().item(), pred_11.mean().item())
    # print("ap:", ap)

    # EO_05
    thresh = 0.5
    y_pred_00 = np.where(pred_00.data.cpu().numpy() >= thresh, 1, 0)
    y_pred_01 = np.where(pred_01.data.cpu().numpy() >= thresh, 1, 0)
    y_pred_10 = np.where(pred_10.data.cpu().numpy() >= thresh, 1, 0)
    y_pred_11 = np.where(pred_11.data.cpu().numpy() >= thresh, 1, 0)
    try:
#         print(y_test_00[:5], y_pred_00[:5])
        tp_target_00 = confusion_matrix(y_test_00.squeeze(), y_pred_00.squeeze()).ravel()[1] / pred_00.shape[0]
#         print(tp_target_00)
        tp_target_01 = confusion_matrix(y_test_01.squeeze(), y_pred_01.squeeze()).ravel()[-1] / pred_01.shape[0]
#         print(tp_target_01)
        tp_target_10 = confusion_matrix(y_test_10.squeeze(), y_pred_10.squeeze()).ravel()[1] / pred_10.shape[0]
#         print(tp_target_10)
        tp_target_11 = confusion_matrix(y_test_11.squeeze(), y_pred_11.squeeze()).ravel()[-1] / pred_11.shape[0]
#         print(tp_target_11)

    except:
        print("Warning:" + "*"*100)
        tp_target_00 = tp_target_01 = tp_target_10 = tp_target_11 = 0
    EO_05 = abs(tp_target_11 - tp_target_01) + abs(tp_target_10 - tp_target_00)
    
    print(f"ap:{ap}, gap_logit:{gap_logit}, EO_05:{EO_05}")

    return ap, gap_logit, EO_05, 0
