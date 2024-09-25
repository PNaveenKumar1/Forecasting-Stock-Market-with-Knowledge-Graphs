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
from models.model_sai import Trans

from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import Sequential, GCNConv, JumpingKnowledge
from torch_geometric.nn import global_mean_pool
from torchmetrics.functional import retrieval_normalized_dcg

from random import randint
import wandb

GPU = 1
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

tau_choices = [5]      #took only one ,that predict after 5 days instead of all 1,5,20
tau_positions = [5]

"""
# Training starts from 2012-01-30
# Testing starts from 2013-04-11
# Phase 1 Test Period - 2013-04-11 to 2013-09-03, 100 days - 0.09 % 6M
# Phase 2 Test Period - 2013-09-04 to 2014-01-27, 100 days - 0.05 % 6M
# Phase 3 Test Period - 2014-01-28 to 2014-06-19, 100 days - 0.07 % 6M
# Phase 4 Test Period - 2014-06-20 to 2014-11-10, 100 days - 0.04 % 6M
# Phase 5 Test Period - 2014-11-12 to 2015-04-07, 100 days - 0.07 % 6M
# Phase 6 Test Period - 2015-04-08 to 2015-08-27, 100 days - 0.117 % 6M
# Phase 7 Test Period - 2015-08-28 to 2016-01-21, 100 days - 0.3016 % 6M
# Phase 8 Test Period - 2016-01-22 to 2016-06-14, 100 days - 0.4279 % 6M
# Phase 9 Test Period - 2016-06-15 to 2016-11-03, 100 days - 0.442 % 6M
# Phase 10 Test Period - 2016-11-04 to 2017-03-30, 100 days - 0.684343434 % 6M
# Phase 11 Test Period - 2017-03-31 to 2017-08-22, 100 days - 1.0689 % 6M
# Phase 12 Test Period - 2017-08-23 to 2018-01-16, 100 days - 1.338181818 % 6M
# Phase 13 Test Period - 2018-01-17 to 2018-06-08, 100 days - 1.9212 % 6M
# Phase 14 Test Period - 2018-06-11 to 2018-10-30, 100 days - 2.271111111 % 6M
# Phase 15 Test Period - 2018-10-31 to 2019-03-27, 100 days - 2.515757576 % 6M
# Phase 16 Test Period - 2019-03-28 to 2019-08-19, 100 days - 2.2441 % 6M
# Phase 17 Test Period - 2019-08-20 to 2020-01-10, 100 days - 1.692857143 % 6M
# Phase 18 Test Period - 2020-01-13 to 2020-06-04, 100 days - 0.6449 % 6M
# Phase 19 Test Period - 2020-06-05 to 2020-10-26, 100 days - 0.134343434 % 6M
# Phase 20 Test Period - 2020-10-27 to 2021-03-22, 100 days - 0.082424242 % 6M
# Phase 21 Test Period - 2021-03-23 to 2021-08-12, 100 days - 0.045544554 % 6M
# Phase 22 Test Period - 2021-08-13 to 2022-01-04, 100 days - 0.082959184 % 6M
# Phase 23 Test Period - 2022-01-05 to 2022-05-27, 100 days - 0.9286 % 6M
# Phase 24 Test Period - 2022-05-31 to 2022-10-20, 100 days - 3.127575758 % 6M
"""
risk_free_returns_in_phase = [0.09, 0.05, 0.07, 0.04, 0.07, 0.117, 0.3016, 0.4279, 0.442, 0.6843,
                            1.0689, 1.3382, 1.9212, 2.2711, 2.5158, 2.2441, 1.6929, 0.6449, 0.1343, 
                            0.0824, 0.0455, 0.0830, 0.9286, 3.1276]
risk_free_returns_in_phase_nifty = [8.4839, 9.0665, 8.8801, 8.6328, 8.2422, 7.7184, 7.2458, 7.0708, 6.6413, 
                                    6.1594, 6.2695, 6.2345, 6.4746, 7.0454, 6.7617, 6.1305, 5.2932, 4.5420,
                                    3.4372, 3.4249, 3.5782, 3.6773, 4.4368, 6.0270]
MODEL = "ours"


FAST = False
if FAST == True:
    LOG = False

if LOG:
    wandb_config = {
        'Prediction Problem': PREDICTION_PROBLEM,
        'Learning Rate':LR,
        'Batch Size': BS,
        'Window Size': W,
        'Prediction Steps': T,
        'Model': MODEL,
        'Optimizer': 'ADAM',
        'Preprocessing': ['Window Divide Min'],
        'KG': ['NO Company'],
        'Transformer Params': [D_MODEL, N_HEAD, DROPOUT, D_FF, ENC_LAYERS, DEC_LAYERS, USE_POS_ENCODING]
    }

    wandb.init(project="KG-Stock-Graph"+str(T), config=wandb_config)

INDEX = ["nasdaq100", "sp500", "nifty500"][0]
print("Experiment {0} With Entire KG P24 W=20 Run 1 RGAT - No rank loss - Adam W - LR 3e-4".format(INDEX))

save_path = "data/pickle/"+INDEX+"/full_graph_data_correct-P25-W"+str(W)+"-T"+str(T)+"_"+str(PREDICTION_PROBLEM)+".pkl"

dataset, company_to_id, graph, hyper_data = load_or_create_dataset_graph(INDEX=INDEX, W=W, T=T, save_path=save_path, problem=PREDICTION_PROBLEM, fast=FAST)

#with open("pickle/"+INDEX+"_meta.pkl", "rb") as f:
    #pickle.dump([company_to_id, graph, hyper_data], f)
    #company_to_id, graph, hyper_data = pickle.load(f)

num_nodes = len(company_to_id.keys())
inverse_company_to_id = {v: k for k, v in company_to_id.items()}

if torch.cuda.is_available():
    device = torch.device("cuda:"+str(GPU))
else:
    device = torch.device("cpu")

if not HYPER_GRAPH:
    graph_nodes_batch = torch.zeros(graph.x.shape[0]).to(device)
    graph = graph.to(device)
    graph_data = {
        'x': graph.x,
        'edge_list': graph.edge_index,
        'batch': graph_nodes_batch
    }
else:
    x, hyperedge_index = hyper_data['x'].to(device), hyper_data['hyperedge_index'].to(device)

    print("Graph details: ", x.shape, hyperedge_index.shape)
    graph_data = {
        'x': x,
        'hyperedge_index': hyperedge_index
    }

relation_graph = None
key = INDEX if INDEX == 'sp500' else INDEX[:-3]
if USE_RELATION_GRAPH == 'gcn':
    file_path = './kg/profile_and_relationship/wikidata/relation_graph.pkl'
    with open(file_path, 'rb') as f:
        relation_graph = pickle.load(f)[key]
    relation_graph = relation_graph.to(device)
elif USE_RELATION_GRAPH == 'hypergraph' or USE_RELATION_GRAPH == 'with_sector':
    file_path = './kg/profile_and_relationship/wikidata/relation_hypergraph.pkl'
    with open(file_path, 'rb') as f:
        relation_graph = pickle.load(f)[key]
    relation_graph = relation_graph.to(device)

kg_file_name = './kg/tkg_create/temporal_kg.pkl'
if INDEX == 'nifty500':
    kg_file_name = './kg/tkg_create/temporal_kg_nifty.pkl'
with open(kg_file_name, 'rb') as f:
    pkl_file = pickle.load(f)

    if "nasdaq" in INDEX:
        kg_map = pkl_file['nasdaq_map']
    elif "sp" in INDEX:
        kg_map = pkl_file['sp_map']
    elif "nifty" in INDEX:
        kg_map = pkl_file['nifty_map']
#print(kg_map)

if USE_KG:
    #kg_file_name = './kg/profile_and_relationship/wikidata/'+INDEX+'_relations_kg.pkl'

    relation_kg = None
    
    """
    kg_file_name = './kg/profile_and_relationship/wikidata/entire_kg.pkl'
    with open(kg_file_name, 'rb') as f:
        pkl_file = pickle.load(f)
        relation_kg = pkl_file['kg']
        if INDEX == 'nasdaq100':
            kg_index = pkl_file['nasdaq_map']
        else:
            kg_index = pkl_file['sp_map']
    
    head, relation, tail = relation_kg[0].long(), relation_kg[1].long(), relation_kg[2].long()
    print(head.max(), relation.max(), tail.max())
    head, relation, tail, kg_index = head.to(device), relation.to(device), tail.to(device), kg_index.to(device)
    relation_kg = (head, relation, tail, kg_index)
    """
else:
    relation_kg = None

if INDEX =='nifty500':
    risk_free_returns_in_phase = risk_free_returns_in_phase_nifty

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


# ----------- Main Training Loop -----------
def predict(loader, desc, kg_map, risk_free_ret):
    epoch_loss = 0

    # TODO: RR is actually RoI (Return on Investment)
    # TODO: Ramit uses some weird formula for RR, which he calls cumulative IRR (Investment Return Ratio)
    # which is not practical, because it's not a ratio, it's a sum of ratio across all assets

    rr, true_rr, accuracy, best_rr, worst_rr = torch.zeros(4).to(device), torch.zeros(4).to(device), torch.zeros(4).to(device), torch.zeros(4).to(device), torch.zeros(4).to(device)
    ndcg, sharpe_ratio = torch.zeros(4).to(device), torch.zeros(4).to(device)
    sharpe = [[], [], []]
    yb_store, yhat_store, yb_store2 = [], [], []

    if model.training and USE_KG and False:
        epoch_kg_loss = 0
        for xb, company, yb, tkg, bestret, worstret in loader:
            head, relation, tail, ts = tkg
            head, relation, tail, ts, kg_map = head.to(device), relation.to(device), tail.to(device), ts.to(device), kg_map.to(device)

            tkg = (head, relation, tail, ts, kg_map)

            kg_loss = model(xb, yb, graph_data, relation_kg, tkg, True)

            kg_loss.backward()
            #nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt_kg.step()
            opt_kg.zero_grad()
            epoch_kg_loss += kg_loss.item()
        print("KG LOSS: ", epoch_kg_loss / len(loader))

    test_rr_list = [[], [], []]
    for xb, yb, tkg, bestret, worstret in loader:
        head, relation, tail, ts = tkg
        head, relation, tail, ts, kg_map = head.to(device), relation.to(device), tail.to(device), ts.to(device), kg_map.to(device)

        tkg = (head, relation, tail, ts, kg_map)

        xb      = xb.to(device)     
        yb      = yb.to(device) 
        bestret = bestret.to(device)
        worstret = worstret.to(device)

        y_hat, kg_loss = model(xb, yb, graph_data, relation_kg, tkg, relation_graph)
        y_hat = y_hat.squeeze()
        # print("before softmax")
        contains_nan = torch.isnan(y_hat).any().item()
        if contains_nan:
            print("The tensor contains NaN values.",contains_nan)
            raise ValueError("Values in the tensor are Nan.")

        y_hat = F.softmax(y_hat.squeeze(), dim = 0)
        #y_hat = torch.sigmoid(y_hat.squeeze())
        true_return_ratio = yb.squeeze() 
        
        true_top5 = torch.topk(true_return_ratio, k=5, dim=0)
        zeros = torch.zeros_like(y_hat)
        zeros[true_top5[1]] = 1

        neg_ret_target_mask = true_return_ratio >= 1
        neg_ret_target = torch.zeros_like(y_hat)
        neg_ret_target[neg_ret_target_mask] = 1
        #print(sum(neg_ret_target))                  # Around half of total assets have negative returns
        #print(y_hat, zeros)
        #loss = F.mse_loss(y_hat, true_return_ratio) #+ rank_loss(y_hat, true_return_ratio)
        # print("before loss1")
        is_in_range = (y_hat >= 0) & (y_hat <= 1)
    
    # Check if all elements in the is_in_range tensor are True
        all_in_range = is_in_range.all().item()
    
        if not all_in_range:
            print("Tensor values:", y_hat)
            raise ValueError("Values in the tensor are outside the range [0, 1].")

        # torch.cuda.synchronize()
        loss = F.binary_cross_entropy(y_hat, zeros) #* 0.3
        # torch.cuda.synchronize()
        #loss += F.binary_cross_entropy(y_hat, zeros2) #* 0.3
        #loss += F.binary_cross_entropy(y_hat, zeros3) #* 0.3
        # print("before loss2")

        loss += F.binary_cross_entropy(y_hat, neg_ret_target)  #* 0.5
        #loss += rank_loss(y_hat, true_return_ratio) # * 0.3  

        tt = torch.argsort(true_return_ratio, descending=True)
        torch.cuda.synchronize()
        rel_score = torch.arange(xb.shape[0], 0, -1).to(device)
        torch.cuda.synchronize()
        true_rel = torch.zeros_like(y_hat).long()
        true_rel[tt] = rel_score
        # print("before loss3")

        loss += approx_ndcg_loss(y_hat.unsqueeze(dim=0), true_rel.unsqueeze(dim=0)) #* 0.3 
        
        epoch_loss += loss.item()     

        if USE_KG or USE_GRAPH == 'hgat':
            loss += kg_loss 
            
        if model.training:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            opt_c.step()
            opt_c.zero_grad()


        for index, k in enumerate(top_k_choice):
            crr, ctrr, cacc, cbr, cwr = evaluate(y_hat[:-1], true_return_ratio, bestret, worstret, k)
            true_rr[index] += ctrr
            rr[index] += crr
            sharpe[index].append(float(crr))
            accuracy[index] += cacc
            best_rr[index] += cbr
            worst_rr[index] += cwr
            ndcg[index] += calculate_ndcg(y_hat, true_return_ratio, k)

            if desc == "TESTING":
                test_rr_list[index].append(float(crr))
        ndcg[3] += calculate_ndcg(y_hat, true_return_ratio, 25)

        if desc == "TESTING":
            plot = torch.topk(y_hat, k=5, dim=0)[1]
            plot2 = true_top5[1]
            plot1_list = []
            plot2_list = []
            for i in range(5):
                plot1_list.append(inverse_company_to_id[plot[i].item()])
                plot2_list.append(inverse_company_to_id[plot2[i].item()])
                #print(inverse_company_to_id[plot[i].item()], inverse_company_to_id[plot2[i].item()], end= " ")
            print("Predicted: ", plot1_list, "Actual: ", plot2_list)
    
    epoch_loss /= len(loader)
    rr /= len(loader) 
    true_rr /= len(loader) 
    accuracy /= len(loader)
    ndcg /= len(loader)
    best_rr /= len(loader)
    worst_rr /= len(loader)

    tau = [252, 252/20]
    for i in range(len(top_k_choice)):
        mean = sum(sharpe[i]) / len(sharpe[i]) * 100
        variance = sum([((x*100 - mean) ** 2) for x in sharpe[i]]) / len(sharpe[i])
        res = (variance*tau[i]) ** 0.5
        sharpe_ratio[i] = (mean*tau[i] - (risk_free_ret)) / (res+0.00001)
        #print(mean, res)

    #print("[{0}] Movement Prediction Accuracy: {1}, MAPE: {2}".format(desc, move_loss.item(), mape.item()))
    #print("[{0}] Range of predictions min: {1} max: {2}".format(desc, mini, maxi))
    #print("[{0}] Epoch: {1} MSE: {2} RMSE: {3} Loss: {4} MAE: {5}".format(desc, ep+1, rmse_returns_loss, rmse_returns_loss ** (1/2), epoch_loss, mae_loss))
    if desc == "TESTING":
        print(test_rr_list)
    print("\n[{0}] Epoch: {1} Loss: {2} NDCG {3}".format(desc, ep+1, epoch_loss, ndcg[3]))
    for index, k in enumerate(top_k_choice):
        print("[{0}] Top {3} NDCG: {5} Return Ratio: {1} True Return Ratio: {2} Accuracy: {4}".format(desc, rr[index], true_rr[index], k, accuracy[index], ndcg[index]))
        print("[{0}] Best RR: {1} Worst RR: {2} Sharpe Ratio: {3}".format(desc, best_rr[index], worst_rr[index], sharpe_ratio[index]))
    #log = {'MSE': epoch_loss, 'RMSE': epoch_loss ** (1/2), "MAPE": mape}
    
    if LOG:
        wandb.log(log)
    PLOT = False
    if PLOT:
        mpl.rcParams['figure.dpi']= 300
        plt.scatter(np.array(yb_store), np.array(yb_store2), c=np.array(yhat_store))
        #plt.plot(np.array(yhat_store).reshape(-1, num_nodes)[:, 0], c='b')
        plt.savefig("plots/saturation/E"+str(ep)+"-T"+str(tau)+ ".png")
        plt.close()
    # print("before returning")

    return epoch_loss, rr, true_rr, accuracy, ndcg, best_rr, worst_rr, sharpe_ratio



for tau in tau_choices:
    tau_pos = tau_positions.index(tau)

    print("Tau: ", tau, "Tau Position: ", tau_pos)

    # ----------- Batching the data -----------
    def collate_fn(instn):
        tkg = instn[0][1]
        instn = instn[0][0]

        # df: shape: Companies x W+1 x 5 (5 is the number of features)
        df = torch.Tensor(np.array([x[0] for x in instn])).unsqueeze(dim=2)
        #df = torch.Tensor(np.array([x[1] for x in instn])).unsqueeze(dim=2) - torch.Tensor(np.array([x[2] for x in instn])).unsqueeze(dim=2)
        for i in range(1, 5):
            df1 = torch.Tensor(np.array([x[i] for x in instn])).unsqueeze(dim=2)
            df = torch.cat((df, df1), dim=2)

        # Shape: Companies x 1
        target = torch.Tensor(np.array([x[7][tau_pos] for x in instn]))

        # Shape: Companies x 1
        #movement = target >= 1

        best_case, worst_case = torch.Tensor(np.array([x[11][tau_pos+1] for x in instn])), torch.Tensor(np.array([x[10][tau_pos+1] for x in instn]))
        best_case = best_case / torch.Tensor(np.array([x[10][0] for x in instn]))
        worst_case = worst_case / torch.Tensor(np.array([x[11][0] for x in instn]))

        return (df, target, tkg, best_case, worst_case)


    start_time, train_begin = 0, 0
    test_mean_rr, test_mean_trr, test_mean_err, test_mean_rrr = [[], [], []], [[], [], []], [[], [], []], [[], [], []]
    test_mean_ndcg, test_mean_acc = [[], [], [], []], [[], [], []]
    test_mean_brr, test_mean_wrr, test_mean_sharpe = torch.zeros(4).to(device), torch.zeros(4).to(device), [[], [], []]

    for phase in range(1, 22):
        print("Phase: ", phase)


        # with open("/data5/home/saihadnoor/Stock_Origin_Kawin/Phase-Stock-KG/pickle/"+INDEX+"_"+str(phase)+"_25phase.pkl", "rb") as f:
        #     # pickle.dump(dataset[train_begin:start_time+400], f)
        #     dataset1 = pickle.load(f)
        #     print(len(dataset1))
        # train_loader    = DataLoader(dataset1, 1, shuffle=True, collate_fn=collate_fn, num_workers=1)
        train_loader    = DataLoader(dataset[train_begin:start_time+400], 1, shuffle=True, collate_fn=collate_fn, num_workers=1)
        val_loader      = DataLoader(dataset[start_time+400:start_time+450], 1, shuffle=False, collate_fn=collate_fn)
        test_loader     = DataLoader(dataset[start_time+450:start_time+550], 1, shuffle=False, collate_fn=collate_fn)   
        #print(len(dataset), len(dataset[start_time:start_time+1000]), len(dataset[start_time+1000:start_time+1100]), len(dataset[start_time+1100:start_time+1400]))
        start_time += 100
        if start_time >= 300:
            train_begin += 100      

        node_type = torch.load('./kg/tkg_create/node_tensor_usa.pt')
        config = {
            'entity_total': 6500,
            'relation_total': 57,
            'L1_flag': False,
            'node_type': node_type,
            'num_node_type': 14
        }

        if INDEX == 'nifty500':
            node_type = torch.load('./kg/tkg_create/node_tensor_india.pt')
            config = {
                'entity_total': 1200,
                'relation_total': 57,
                'L1_flag': False,
                'node_type': node_type,
                'num_node_type': 16
            }
        model  = Trans(W, T, D_MODEL, N_HEAD, ENC_LAYERS, DEC_LAYERS, D_FF, DROPOUT, USE_POS_ENCODING, USE_GRAPH, HYPER_GRAPH, USE_KG, num_nodes, config, "transf", USE_RELATION_GRAPH)
        if phase == 1:
            print(model)
        model.to(device)

        #if phase > 1:
        #    model.load_state_dict(torch.load("models/saved_models/best_model_"+INDEX+str(W)+"_"+str(T)+"_"+str(RUN)+".pt"))
        #nasdaq 1e-5, 4e-5
        opt_c = torch.optim.AdamW(model.parameters(), lr = 2e-5, betas=(0.9, 0.999), eps=1e-08)
        opt_kg = torch.optim.Adam(model.parameters(), lr = 4e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        #opt_c = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0, nesterov=True)

        prev_val_loss, best_val_loss = float("infinity"), float("infinity")
        val_loss_history = []
        for ep in range(4):
            print("Epoch: " + str(ep+1))

            #if MODEL_TYPE == 'random':
            #    continue
            
            model.train()
            train_epoch_loss, rr, trr, accuracy, ndcg, bestr, worstr, sharpe = predict(train_loader, "TRAINING", kg_map, risk_free_returns_in_phase[phase-1])

            model.eval()
            with torch.no_grad():
                val_epoch_loss, rr, trr, accuracy, ndcg, bestr, worstr, sharpe = predict(val_loader, "VALIDATION", kg_map, risk_free_returns_in_phase[phase-1])

            #plot(val_loader)

            if prev_val_loss < val_epoch_loss:
                val_loss_history.append(1)
            else:
                val_loss_history.append(0)
            prev_val_loss = val_epoch_loss
            
            if best_val_loss >= val_epoch_loss: # and (ep > MAX_EPOCH//2 or ep > 15):
                print("Saving Model")
                #if USE_GRAPH == 'hgat':
                    #torch.save(model.tsne_emb, "results/emb/"+ INDEX + str(phase)+ "_"+str(tau)+ "_tsne_emb.pt")
                    #torch.save(model.tsne_emb_all, "results/emb/"+ INDEX + str(phase)+ "_"+str(tau)+ "_tsne_emb_all.pt")
                    #torch.save(model.month_embeddings, "results/emb/"+INDEX + str(phase)+ "_"+str(tau)+ "_month_emb.pt")
                    #torch.save(model.hour_embeddings,"results/emb/"+ INDEX + str(phase)+ "_"+str(tau)+ "_hour_emb.pt")
                    #torch.save(model.day_embeddings, "results/emb/"+INDEX + str(phase)+ "_"+str(tau)+ "_day_emb.pt")
                    #torch.save(model.rel_type_embeddings, "results/emb/"+INDEX + str(phase)+ "_"+str(tau)+ "_reltype_emb.pt")
                torch.save(model.state_dict(), "models/saved_models/best_model_"+INDEX+str(W)+"_"+str(T)+"_"+str(RUN)+".pt")
                best_val_loss = val_epoch_loss
            
            if ep > 7 and sum(val_loss_history[-3:]) == 3:
                print("Early Stopping")
                break

        if MODEL_TYPE != 'random':
            model.load_state_dict(torch.load("models/saved_models/best_model_"+INDEX+str(W)+"_"+str(T)+"_"+str(RUN)+".pt"))

        model.eval()
        with torch.no_grad():
            test_epoch_loss, rr, trr, accuracy, ndcg, bestr, worstr, sharpe  = predict(test_loader, "TESTING", kg_map, risk_free_returns_in_phase[phase-1])
            for i in range(len(top_k_choice)):
                test_mean_rr[i].append(rr[i].item())
                test_mean_trr[i].append(trr[i].item())
                test_mean_ndcg[i].append(ndcg[i].item())
                test_mean_acc[i].append(accuracy[i].item())
                test_mean_sharpe[i].append(sharpe[i].item())
            test_mean_ndcg[3].append(ndcg[3].item())
            test_mean_brr += bestr
            test_mean_wrr += worstr
            print("NDCG: mean {0} std {1}".format(sum(test_mean_ndcg[3])/phase, np.std(np.array(test_mean_ndcg[3]))))
            for index, k in enumerate(top_k_choice):
                print("[Mean - {0}] Top {3} NDCG: {5} Return Ratio: {1} True Return Ratio: {2} Accuracy: {4}".format("TESTING", sum(test_mean_rr[index])/phase, sum(test_mean_trr[index])/phase, k, sum(test_mean_acc[index])/phase, sum(test_mean_ndcg[index])/phase))
                print("[Mean - {0}] Best Return Ratio: {1} Worst Return Ratio: {2} Sharpe Ratio: {3}".format("TESTING", test_mean_brr[index]/phase, test_mean_wrr[index]/phase, sum(test_mean_sharpe[index])/phase))
                print("[STD - {0}] Top {3} NDCG: {5} Return Ratio: {1} True Return Ratio: {2} Accuracy: {4} Sharpe: {6}".format("TESTING", np.std(np.array(test_mean_rr[index])), np.std(np.array(test_mean_trr[index])), k, np.std(np.array(test_mean_acc[index])), np.std(np.array(test_mean_ndcg[index])), np.std(np.array(test_mean_sharpe[index]))))

            #print("[Mean - {0}] Movement Accuracy: {1} Mean MAPE: {2} Mean MAE: {3}".format("TESTING", test_mean_move/phase, test_mean_mape/phase, test_mean_mae/phase))
        if LOG:
            wandb.save('model.py')

    # phase = 14
    print("Tau: ", tau)
    for index, k in enumerate(top_k_choice):
        print("[Result Copy] Top {1} {2} {3} {4}".format("TESTING", k, sum(test_mean_ndcg[index])/phase, sum(test_mean_acc[index])/phase, sum(test_mean_rr[index])/phase))
        print("[Result Copy] Top {1} {2} {3} {4}".format("TESTING", k, sum(test_mean_sharpe[index])/phase, test_mean_brr[index]/phase, test_mean_wrr[index]/phase))
    #print("[Mean - {0}] {1} {2} {3}".format("TESTING", test_mean_move/phase, test_mean_mape/phase, test_mean_mae/phase))
    



