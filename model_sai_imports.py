import csv
import os, sys
import pickle
import math
from queue import PriorityQueue
from tkinter import N

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import pickle

mpl.rcParams['figure.dpi']= 300

from sklearn.metrics import accuracy_score, ndcg_score
#from torchinfo import summary
from tqdm import tqdm
from utils import (mean_absolute_percentage_error,
                   load_or_create_dataset_graph,
                   mean_square_error, root_mean_square_error)

# from models.models import Transformer_Ranking, Saturation
# from models.model_sai import Trans
# from models.model_trans_dec import Trans
# from models.model_trans_Enc import Trans
from models.model_nbeats import Trans

from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import Sequential, GCNConv, JumpingKnowledge
from torch_geometric.nn import global_mean_pool
from torchmetrics.functional import retrieval_normalized_dcg

from random import randint
import wandb

GPU = 8
LR = 0.0006
BS = 128
W = 20
T = 20
LOG = False
D_MODEL = 20
N_HEAD  = 5
DROPOUT = 0.1
D_FF    = 1024
ENC_LAYERS = 1
DEC_LAYERS = 1
MAX_EPOCH = 10
USE_POS_ENCODING = False
USE_GRAPH = [True, False, 'hgat', 'rgat', 'gcn'][2]
HYPER_GRAPH = [True, False][1]
USE_RELATION_GRAPH = ['gcn', 'hypergraph', 'with_sector', False][3]
USE_KG = [True, False][1]
PREDICTION_PROBLEM = 'value'
RUN = randint(1, 100000)
PLOT = False
MODEL_TYPE = '' #'random'
ENCODER_LAYER = ['gru', 'transf', 'lstm'][2]

# device = torch.device("cuda:"+str(GPU))
if torch.cuda.is_available():
    device_idx = torch.cuda.current_device()
    device = torch.device("cuda:"+str(device_idx))

# tau_choices = [5]      #took only one ,that predict after 5 days instead of all 1,5,20
# tau_positions = [5]
tau_choices = [1,5,20]
tau_positions = [1, 5, 20]
FAST = False
risk_free_returns_in_phase = [0.09, 0.05, 0.07, 0.04, 0.07, 0.117, 0.3016, 0.4279, 0.442, 0.6843,
                            1.0689, 1.3382, 1.9212, 2.2711, 2.5158, 2.2441, 1.6929, 0.6449, 0.1343, 
                            0.0824, 0.0455, 0.0830, 0.9286, 3.1276]
risk_free_returns_in_phase_nifty = [8.4839, 9.0665, 8.8801, 8.6328, 8.2422, 7.7184, 7.2458, 7.0708, 6.6413, 
                                    6.1594, 6.2695, 6.2345, 6.4746, 7.0454, 6.7617, 6.1305, 5.2932, 4.5420,
                                    3.4372, 3.4249, 3.5782, 3.6773, 4.4368, 6.0270]
MODEL = "ours"
def rank_loss(prediction, ground_truth):
    all_one = torch.ones(prediction.shape[0], 1, dtype=torch.float32).to(device)
    prediction = prediction.unsqueeze(dim=1)
    ground_truth = ground_truth.unsqueeze(dim=1)
    #print(prediction.shape, ground_truth.shape, base_price.shape)
    return_ratio = prediction 
    true_return_ratio = ground_truth - 1

    pre_pw_dif = torch.sub(
        return_ratio @ all_one.t(),                  # C x C
        all_one @ return_ratio.t()                   # C x C
    )
    gt_pw_dif = torch.sub(
        all_one @ true_return_ratio.t(),
        true_return_ratio @ all_one.t()
    )

    rank_loss = torch.mean(
        F.relu(-1*pre_pw_dif * gt_pw_dif )
    )
   
    return rank_loss 

def evaluate(prediction, ground_truth, bestret, worstret, K):
    return_ratio = prediction - 1
    true_return_ratio = ground_truth - 1
    bestret = bestret - 1
    worstret = worstret - 1

    #print("True top k: ", torch.topk(true_return_ratio.squeeze(), k=3, dim=0))
    #print("Predicted top k: ", torch.topk(return_ratio.squeeze(), k=3, dim=0))

    target_obtained_return_ratio = torch.topk(true_return_ratio, k=K, dim=0)[0].mean()

    #random = torch.randint(0, prediction.shape[0]-1, (K,))
    #random_return_ratio = true_return_ratio[random].mean()
    
    topk_predicted = torch.topk(return_ratio, k=K, dim=0)[1]

    #global MODEL_TYPE
    #if MODEL_TYPE == 'random':
    #    topk_predicted = random

    obtained_return_ratio = true_return_ratio[topk_predicted].mean()
    best_return_ratio = bestret[topk_predicted].mean()
    worst_return_ratio = worstret[topk_predicted].mean()
    #return_ratio = -1*return_ratio
    #obtained_return_ratio += true_return_ratio[torch.topk(return_ratio.squeeze(), k=K, dim=0)[1]].mean()
    #obtained_return_ratio /= 2

    a_cat_b, counts = torch.cat([torch.topk(return_ratio.squeeze(), k=K, dim=0)[1], torch.topk(true_return_ratio.squeeze(), k=K, dim=0)[1]]).unique(return_counts=True)
    accuracy = a_cat_b[torch.where(counts.gt(1))].shape[0] / K

    return obtained_return_ratio, target_obtained_return_ratio, accuracy, best_return_ratio, worst_return_ratio


top_k_choice = [1, 5]

def calculate_ndcg(predict, true, k):
    tt = torch.topk(true, k, dim=0)[1]
    rel_score = torch.arange(k, 0, -1).to(device)
    true_rel = torch.zeros_like(predict).long()
    true_rel[tt] = rel_score
    return retrieval_normalized_dcg(predict, true_rel)

def approxNDCGLoss(y_pred, y_true, eps=1e-10, padded_value_indicator=-1, alpha=1.):
    """
    Loss based on approximate NDCG introduced in "A General Approximation Framework for Direct Optimization of
    Information Retrieval Measures". Please note that this method does not implement any kind of truncation.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param alpha: score difference weight used in the sigmoid function
    :return: loss value, a torch.Tensor
    """
    # shuffle for randomised tie resolution
    random_indices = torch.randperm(y_pred.shape[-1])
    y_pred_shuffled = y_pred.unsqueeze(dim=0)[:, random_indices]
    y_true_shuffled = y_true.unsqueeze(dim=0)[:, random_indices]

    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

    mask = y_true_sorted == padded_value_indicator

    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
    preds_sorted_by_true[mask] = float("-inf")

    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

    observation_loss[mask] = 0.0

    return torch.mean(torch.sum(observation_loss, dim=1))

def approx_rank(logits):
    """_summary_

    Args:
        logits (_type_): A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.

    Returns:
        _type_: A `Tensor` of ranks with the same shape as logits.
    """
    list_size = logits.shape[1]
    x = logits.unsqueeze(2).repeat(1, 1, list_size)
    y = logits.unsqueeze(1).repeat(1, list_size, 1)
    rank = torch.sigmoid(x - y)
    rank = torch.sum(rank, dim=-1) #+ 0.5
    return rank

def approx_ndcg_loss(logits, labels):
    """_summary_

    Args:
        logits (_type_): A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.
        labels (_type_): A `Tensor` with shape [batch_size, list_size]. Each value is the
      relevance label of the corresponding item.

    Returns:
        _type_: A `Tensor` of ndcg loss with shape [batch_size].
    """
    rank = approx_rank(logits)
    #print("logits", torch.topk(logits, k=5, dim=-1), torch.topk(rank, k=5, dim=-1))
    return - retrieval_normalized_dcg(rank, labels)

