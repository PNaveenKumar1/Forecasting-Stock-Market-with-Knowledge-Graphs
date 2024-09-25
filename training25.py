import model_sai_imports
from model_sai_imports import *
from torch import autograd
from model_sai_optim import get_dataset

dataset = get_dataset()
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
        # print("hellop",file=sys.stderr)
        # df: shape: Companies x W+1 x 5 (5 is the number of features)
        df = torch.Tensor(np.array([x[0] for x in instn])).unsqueeze(dim=2)
        #df = torch.Tensor(np.array([x[1] for x in instn])).unsqueeze(dim=2) - torch.Tensor(np.array([x[2] for x in instn])).unsqueeze(dim=2)
        for i in range(1, 5):
            df1 = torch.Tensor(np.array([x[i] for x in instn])).unsqueeze(dim=2)
            df = torch.cat((df, df1), dim=2)
        min_val = df.min()
        max_val = df.max()

        # Normalize tensor to the range [-1, 1]
        normalized_tensor = 2 * (df - min_val) / (max_val - min_val) - 1
        # Shape: Companies x 1
        target = torch.Tensor(np.array([x[7][tau_pos] for x in instn]))

        # Shape: Companies x 1
        #movement = target >= 1

        best_case, worst_case = torch.Tensor(np.array([x[11][tau_pos+1] for x in instn])), torch.Tensor(np.array([x[10][tau_pos+1] for x in instn]))
        best_case = best_case / torch.Tensor(np.array([x[10][0] for x in instn]))
        worst_case = worst_case / torch.Tensor(np.array([x[11][0] for x in instn]))

        return (normalized_tensor, target, tkg, best_case, worst_case)
    for phase in range(1,25):
        print("Phase: ", phase)
        train_loader    = DataLoader(dataset[train_begin:start_time+400], 1, shuffle=True, collate_fn=collate_fn, num_workers=1)
        val_loader      = DataLoader(dataset[start_time+400:start_time+450], 1, shuffle=False, collate_fn=collate_fn)
        test_loader     = DataLoader(dataset[start_time+450:start_time+550], 1, shuffle=False, collate_fn=collate_fn)   
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
        
