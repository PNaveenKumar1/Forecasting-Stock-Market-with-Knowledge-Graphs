import model_sai_imports
from model_sai_imports import *
from torch import autograd
INDEX = ["nasdaq100", "sp500", "nifty500"][0]
print("Experiment {0} With Entire KG P24 W=20 Run 1 RGAT - No rank loss - Adam W - LR 3e-4".format(INDEX))

import wandb

# Initialize wandb
# wandb.init(project='stock_2_with_log_TRANS_DEC')

if torch.cuda.is_available():
    device_idx = torch.cuda.current_device()
    device = torch.device("cuda:"+str(device_idx))
    print(torch.cuda.get_device_name())
else:
    device = torch.device("cpu")

save_path = "data/pickle/"+INDEX+"/full_graph_data_correct-P25-W"+str(W)+"-T"+str(T)+"_"+str(PREDICTION_PROBLEM)+".pkl"

dataset, company_to_id, graph, hyper_data = load_or_create_dataset_graph(INDEX=INDEX, W=W, T=T, save_path=save_path, problem=PREDICTION_PROBLEM, fast=FAST)
num_nodes = len(company_to_id.keys())
inverse_company_to_id = {v: k for k, v in company_to_id.items()}
# device = 0
# if torch.cuda.is_available():
#     device = torch.device("cuda:"+str(GPU))
# else:
#     device = torch.device("cpu")

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
def predict(loader, desc, kg_map, risk_free_ret):
    epoch_loss = 0

    # TODO: RR is actually RoI (Return on Investment)
    # TODO: Ramit uses some weird formula for RR, which he calls cumulative IRR (Investment Return Ratio)
    # which is not practical, because it's not a ratio, it's a sum of ratio across all assets

    rr, true_rr, accuracy, best_rr, worst_rr = torch.zeros(4).to(device), torch.zeros(4).to(device), torch.zeros(4).to(device), torch.zeros(4).to(device), torch.zeros(4).to(device)
    ndcg, sharpe_ratio = torch.zeros(4).to(device), torch.zeros(4).to(device)
    sharpe = [[], [], []]
    yb_store, yhat_store, yb_store2 = [], [], []

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
        y_hat = F.softmax(y_hat.squeeze(), dim = 0)
        true_return_ratio = yb.squeeze() 
        
        true_top5 = torch.topk(true_return_ratio, k=5, dim=0)
        zeros = torch.zeros_like(y_hat)
        zeros[true_top5[1]] = 1

        neg_ret_target_mask = true_return_ratio >= 1
        neg_ret_target = torch.zeros_like(y_hat)
        neg_ret_target[neg_ret_target_mask] = 1
        loss = F.binary_cross_entropy(y_hat, zeros) #* 0.3
        loss += F.binary_cross_entropy(y_hat, neg_ret_target)  #* 0.5
        tt = torch.argsort(true_return_ratio, descending=True)
        rel_score = torch.arange(xb.shape[0], 0, -1).to(device)
        true_rel = torch.zeros_like(y_hat).long()
        true_rel[tt] = rel_score
        loss += approx_ndcg_loss(y_hat.unsqueeze(dim=0), true_rel.unsqueeze(dim=0)) #* 0.3 
        epoch_loss += loss.item()     
        if USE_KG or USE_GRAPH == 'hgat':
            loss += kg_loss 
            
        if model.training:
            # with autograd.detect_anomaly():
            loss.backward()
            clipped_gradient_norm=nn.utils.clip_grad_norm_(model.parameters(),4.0)
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
    log = {'MSE': epoch_loss}
    
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

    start_time, train_begin = 0, 0
    test_mean_rr, test_mean_trr, test_mean_err, test_mean_rrr = [[], [], []], [[], [], []], [[], [], []], [[], [], []]
    test_mean_ndcg, test_mean_acc = [[], [], [], []], [[], [], []]
    test_mean_brr, test_mean_wrr, test_mean_sharpe = torch.zeros(4).to(device), torch.zeros(4).to(device), [[], [], []]
    def collate_fn(instn):
        tkg = instn[0][1]
        instn = instn[0][0]
        df = torch.Tensor(np.array([x[0] for x in instn])).unsqueeze(dim=2)
        for i in range(1, 5):
            df1 = torch.Tensor(np.array([x[i] for x in instn])).unsqueeze(dim=2)
            df = torch.cat((df, df1), dim=2)
        min_val = df.min()
        max_val = df.max()

        # Normalize tensor to the range [-1, 1]
        normalized_tensor = 2 * (df - min_val) / (max_val - min_val) - 1
        target = torch.Tensor(np.array([x[7][tau_pos] for x in instn]))

        best_case, worst_case = torch.Tensor(np.array([x[11][tau_pos+1] for x in instn])), torch.Tensor(np.array([x[10][tau_pos+1] for x in instn]))
        best_case = best_case / torch.Tensor(np.array([x[10][0] for x in instn]))
        worst_case = worst_case / torch.Tensor(np.array([x[11][0] for x in instn]))

        return (normalized_tensor, target, tkg, best_case, worst_case)
    for phase in range(1,8):
        print("Phase: ", phase)
        # train_loader    = DataLoader(dataset[start_time:start_time+1000], 1, shuffle=True, collate_fn=collate_fn, num_workers=1)
        # val_loader      = DataLoader(dataset[start_time+1000:start_time+1100], 1, shuffle=False, collate_fn=collate_fn)
        # test_loader     = DataLoader(dataset[start_time+1100:start_time+1400], 1, shuffle=False, collate_fn=collate_fn)   
        train_loader    = DataLoader(dataset[start_time:start_time+750], 1, shuffle=True, collate_fn=collate_fn, num_workers=1)
        val_loader      = DataLoader(dataset[start_time+750:start_time+825], 1, shuffle=False, collate_fn=collate_fn)
        test_loader     = DataLoader(dataset[start_time+825:start_time+1025], 1, shuffle=False, collate_fn=collate_fn)
        # train_loader    = DataLoader(dataset[train_begin:start_time+400], 1, shuffle=True, collate_fn=collate_fn, num_workers=1)
        # val_loader      = DataLoader(dataset[start_time+400:start_time+450], 1, shuffle=False, collate_fn=collate_fn)
        # test_loader     = DataLoader(dataset[start_time+450:start_time+550], 1, shuffle=False, collate_fn=collate_fn)        
         
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
        model  = Trans(W, T, D_MODEL, N_HEAD, ENC_LAYERS, DEC_LAYERS, D_FF, DROPOUT, USE_POS_ENCODING, USE_GRAPH, HYPER_GRAPH, USE_KG, num_nodes, config, "transf", USE_RELATION_GRAPH)
        if phase == 1:
            print(model)
        model.to(device)

        #if phase > 1:
        #    model.load_state_dict(torch.load("models/saved_models/best_model_"+INDEX+str(W)+"_"+str(T)+"_"+str(RUN)+".pt"))
        #nasdaq 1e-5, 4e-5
        opt_c = torch.optim.AdamW(model.parameters(), lr = 1e-5, betas=(0.9, 0.999), eps=1e-08)
        opt_kg = torch.optim.Adam(model.parameters(), lr = 4e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        # opt_c = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0, nesterov=True)

        prev_val_loss, best_val_loss = float("infinity"), float("infinity")
        val_loss_history = []
    
        for ep in range(10):
            print("Epoch: " + str(ep+1))  
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

                # model.eval()
            #print("[Mean - {0}] Movement Accuracy: {1} Mean MAPE: {2} Mean MAE: {3}".format("TESTING", test_mean_move/phase, test_mean_mape/phase, test_mean_mae/phase))
        if LOG:
            wandb.save('model.py')
    print("Tau: ", tau)
    for index, k in enumerate(top_k_choice):
        print("[Result Copy] Top {1} {2} {3} {4}".format("TESTING", k, sum(test_mean_ndcg[index])/phase, sum(test_mean_acc[index])/phase, sum(test_mean_rr[index])/phase))
        print("[Result Copy] Top {1} {2} {3} {4}".format("TESTING", k, sum(test_mean_sharpe[index])/phase, test_mean_brr[index]/phase, test_mean_wrr[index]/phase))
        #print("[Mean - {0}] {1} {2} {3}".format("TESTING", test_mean_move/phase, test_mean_mape/phase, test_mean_mae/phase))
        

   

