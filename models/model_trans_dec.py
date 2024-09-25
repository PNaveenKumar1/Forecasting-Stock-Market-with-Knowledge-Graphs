import torch
import torch.nn as nn 
import math 
import torch.nn.functional as F
from torch_geometric.nn import Sequential, GCNConv, JumpingKnowledge, HypergraphConv, RGATConv, HEATConv ,HANConv,GATConv
from torch_geometric.nn import global_mean_pool
from torchkge.models.translation import TorusEModel
from torchkge.models.bilinear import ComplExModel, HolEModel
from models.tkge_models import *
from torch_geometric.nn import global_mean_pool
import torch_geometric.nn as gnn

class Trans(nn.Module):
    
    def __init__(self, W, T, D_MODEL, N_HEAD, ENC_LAYERS, DEC_LAYERS, D_FF, DROPOUT, USE_POS_ENCODING = False, USE_GRAPH = False, HYPER_GRAPH = True, USE_KG = True, NUM_NODES = 87, config=None, ENCODER_LAYER = 'transf', USE_RELATION_KG = False):
        super().__init__()
        SEC_EMB, n = 25, 0 # 1 For LSTM Embedding
        if USE_GRAPH:
            if HYPER_GRAPH:
                n += 0
            n += 2
        if USE_KG:
            n += 2
        if USE_RELATION_KG:
            n += 1
        config['embedding_size'] = SEC_EMB*2

        self.embeddings = nn.Embedding(105, 10)
        
        self.encoder_architecture = 'transf'
        if self.encoder_architecture == 'transf':
            decoder_layer = nn.TransformerDecoderLayer(d_model=D_MODEL, nhead=N_HEAD,dim_feedforward=D_FF,batch_first=True)
            self.transformer_decoder_first = nn.TransformerDecoder(decoder_layer, num_layers=1)
            encoder_layer = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=N_HEAD, dim_feedforward=D_FF,batch_first=True)
            self.transformer_encoder_first = nn.TransformerEncoder(encoder_layer, num_layers=ENC_LAYERS)
        else:
            raise NotImplementedError("Encoder Architecture not implemented. Choose between [transf, lstm, gru]")
        
        self.fc1 = nn.Linear(D_MODEL, D_FF)
        self.fc2 = nn.Linear(D_FF, D_MODEL)
        self.pred = nn.Linear((D_MODEL+(SEC_EMB*n))*NUM_NODES, NUM_NODES)
        # self.pred = nn.Linear(20, NUM_NODES)

        self.pred2 = nn.Linear(NUM_NODES*10, NUM_NODES)

        self.hold_pred = nn.Linear(D_MODEL+(SEC_EMB*n), 1)

        self.is_pos = USE_POS_ENCODING
        self.time_steps = T

        self.use_graph = USE_GRAPH
        self.is_hyper_graph = HYPER_GRAPH
        if self.use_graph:# == True:
            if self.is_hyper_graph:
                self.graph_model = Sequential('x, hyperedge_index', [
                        #(Dropout(p=0.5), 'x -> x'),
                        (HypergraphConv(8, 32, dropout=0.1), 'x, hyperedge_index -> x1'),
                        nn.LeakyReLU(inplace=True),
                        (HypergraphConv(32, 32, dropout=0.1), 'x1, hyperedge_index -> x2'),
                        nn.LeakyReLU(inplace=True),
                        
                        #(lambda x1, x2: [x1, x2], 'x1, x2 -> xs'),
                        #(JumpingKnowledge("cat", 64, num_layers=2), 'xs -> x'),
                        #(global_mean_pool, 'x, batch -> x'),
                        nn.Linear(32, SEC_EMB*2),
                    ])
            elif self.use_graph == 'gcn':
                self.graph_model = Sequential('x, edge_index, batch', [
                            #(Dropout(p=0.5), 'x -> x'),
                            (GCNConv(20, 64), 'x, edge_index -> x1'),
                            nn.ReLU(inplace=True),
                            (GCNConv(64, 64), 'x1, edge_index -> x2'),
                            nn.ReLU(inplace=True),
                            nn.Linear(64, SEC_EMB*2),
                        ])
        self.use_relation_graph = USE_RELATION_KG
        if self.use_relation_graph == 'gcn':
            self.rel_node_emb = nn.Embedding(5000, 8)
            self.rel_graph_model = Sequential('x, edge_index, batch', [
                            #(Dropout(p=0.5), 'x -> x'),
                            (GCNConv(8, 32), 'x, edge_index -> x1'),
                            nn.ReLU(inplace=True),
                            (GCNConv(32, 64), 'x1, edge_index -> x2'),
                            nn.ReLU(inplace=True),
                            nn.Linear(64, SEC_EMB),
                        ])
        elif self.use_relation_graph == 'hypergraph' or self.use_relation_graph == 'with_sector':
            self.rel_node_emb = nn.Embedding(5000, 8)
            self.rel_graph_model = Sequential('x, hyperedge_index', [
                        #(Dropout(p=0.5), 'x -> x'),
                        (HypergraphConv(8, 32, dropout=0.1), 'x, hyperedge_index -> x1'),
                        nn.LeakyReLU(inplace=True),
                        (HypergraphConv(32, 32, dropout=0.1), 'x1, hyperedge_index -> x2'),
                        nn.LeakyReLU(inplace=True),
                        nn.Linear(32, SEC_EMB),
                    ])

        if self.use_graph == 'rgat':
            self.conv1 = RGATConv(20, 32, config['relation_total'], edge_dim=16, heads=1, dim=1)#, mod="additive")
            #self.global_pool = aggr.MaxAggregation()
            self.conv2 = RGATConv(32, 64, config['relation_total'], edge_dim=16, heads=1, dim=1)#, mod="additive")
            #self.global_pool2 = aggr.MaxAggregation()
            #self.conv3 = RGATConv(64, 64, config['relation_total'], edge_dim=16, heads=1, dim=1)
            self.lin = nn.Linear(64, SEC_EMB*2)
            self.lin2 = nn.Linear(SEC_EMB*2, 16)

            self.rel_type_embeddings = nn.Embedding(config['relation_total']+2, 16)
            #self.year_embeddings    = nn.Embedding(24, 16, padding_idx=0)
            self.month_embeddings   = nn.Embedding(13, 16, padding_idx=0)
            self.day_embeddings     = nn.Embedding(32, 16, padding_idx=0)
            self.hour_embeddings    = nn.Embedding(25, 16, padding_idx=0)
            self.minutes_embeddings = nn.Embedding(61, 16, padding_idx=0)
            self.sec_embeddings     = nn.Embedding(61, 16, padding_idx=0)
        if self.use_graph == 'hgat':
            self.node_type = config['node_type']
            self.conv1 = HEATConv(20, 8, config['num_node_type'], config['relation_total'], 16, 16, SEC_EMB*2, 4, dropout=0.1)
            # self.conv1 = GATConv(20, 8,dropout=0.1)

            #self.global_pool = aggr.MaxAggregation()
            #self.bn1 = gnn.norm.BatchNorm(32)
            self.conv2 = HEATConv(8*4, 16, config['num_node_type'], config['relation_total'], 16, 16, SEC_EMB*2, 4, dropout=0.1)
            
            # self.conv2 = GATConv(8*4, 16, dropout=0.1)

            #self.global_pool2 = aggr.MaxAggregation()
            #self.bn2 = gnn.norm.BatchNorm(64)
            self.lin = nn.Linear(64, SEC_EMB*2)
            self.lin2 = nn.Linear(SEC_EMB*2, 16)
            self.num_rel = config['relation_total']
            self.rel_type_embeddings = nn.Embedding(config['relation_total']+2, 16)
            #self.year_embeddings    = nn.Embedding(24, 16, padding_idx=0)
            self.month_embeddings   = nn.Embedding(13, 16, padding_idx=0)
            self.day_embeddings     = nn.Embedding(32, 16, padding_idx=0)
            self.hour_embeddings    = nn.Embedding(25, 16, padding_idx=0)
            self.minutes_embeddings = nn.Embedding(61, 16, padding_idx=0)
            self.sec_embeddings     = nn.Embedding(61, 16, padding_idx=0)
                
        self.use_kg = USE_KG
        self.config = config

        if self.use_kg:
            #self.relation_kge = TorusEModel(n_entities= 5500, n_relations = 40, emb_dim = SEC_EMB, dissimilarity_type='torus_L2')
            #self.relation_kge = HolEModel(n_entities= config['entity_total'], n_relations = config['relation_total'], emb_dim = config['embedding_size'] )
             
            self.kge = TTransEModel(config) #TADistmultModel(config) #TTransEModel(config)
        if self.use_kg:
            pass
            #self.temporal_kge = TorusEModel(n_entities= 5500, n_relations = 50, emb_dim = SEC_EMB, dissimilarity_type='torus_L2')
            #self.temporal_kge = HolEModel(n_entities= 5500, n_relations = 40, emb_dim = SEC_EMB) 
        self.num_nodes = NUM_NODES

        self.ent_embeddings = nn.Embedding(config['entity_total']+2, 20)
        self.num_nodes = config['entity_total']+2
 
    def forward(self, xb, yb=None, graph=None, kg=None, tkg=None, relation_graph = None):
        kg_loss = 0
        if self.encoder_architecture == 'transf':
            # print(xb.shape)
            x_t= xb.transpose(1,2)
            x_enc = self.transformer_encoder_first(x_t)
            batch_size,sequence_length,embedding_dim = x_enc.size()
            target_length = 83  #442 336 83
            device = 0
            if torch.cuda.is_available():
                device_idx = torch.cuda.current_device()
                device = torch.device("cuda:"+str(device_idx))
                # device = torch.device("cuda:"+str(8))
            # memory_mask = torch.tril(torch.ones(target_length, sequence_length, dtype=torch.bool)).to(device)
            memory_mask = torch.zeros(target_length, sequence_length, dtype=torch.bool).to(device)
            contains_nan = torch.isnan(memory_mask).any().item()
            if contains_nan:
                print("The tensor contains NaN values.",contains_nan)
                raise ValueError("Values in the tensor are Nan.")
            dec_out = self.transformer_decoder_first(tgt=x_t, memory=x_enc, memory_key_padding_mask=memory_mask).mean(dim=1)
            # dec_out = self.transformer_decoder_first(tgt=x_t , memory= x_enc).mean(dim=1)
            x = dec_out.unsqueeze(dim=0)
            print(x.shape)
            contains_nan = torch.isnan(dec_out).any().item()
            if contains_nan:
                print("The tensor contains NaN values.",contains_nan)
                print(dec_out)
                raise ValueError("Values in the tensor are Nan.")

            # x = self.transformer_encoder_first(xb)
        if self.use_graph == 'gcn' and not self.is_hyper_graph:
            print("we are at 2")
            edge = torch.cat((tkg[0].unsqueeze(0), tkg[2].unsqueeze(0)), dim=0)
            batch = torch.ones(edge.shape[1]).long().to(tkg[0].device)
            g_emb = self.graph_model(self.ent_embeddings.weight, edge, batch)[tkg[4].long()]
            g_emb = g_emb.unsqueeze(dim=0)            
            
            #g_emb = self.graph_model(graph['x'], graph['edge_list'], graph['batch'])    
            x = torch.cat((x, g_emb), dim=2)



        if self.use_graph and self.is_hyper_graph:
            print("we are at 1")
            g_emb = self.graph_model(graph['x'], graph['hyperedge_index']).unsqueeze(dim=0)
            #g_emb = g_emb.repeat(1, self.time_steps, 1)
            x = torch.cat((x, g_emb), dim=2)
        if self.use_graph == 'hgat':
            # print("we are at 6")
            edge = torch.cat((tkg[0].unsqueeze(0), tkg[2].unsqueeze(0)), dim=0) 
            batch = torch.ones(self.num_nodes).long().to(tkg[0].device)
            node_type = self.node_type.to(tkg[0].device)
            
            temp_emb = self.month_embeddings(tkg[3][:, 1]) + \
                    self.day_embeddings(tkg[3][:, 2]) + self.hour_embeddings(tkg[3][:, 3]) 
            rel_emb = self.rel_type_embeddings(tkg[1])
            edge_attr = temp_emb + rel_emb

            weight = self.ent_embeddings.weight.clone()
            weight[tkg[4].long()] = self.ent_embeddings.weight[tkg[4].long()] + x[0]
            print("weight shape",weight.shape)
            print("edge shape",edge.shape)  
            print("node_type shape",node_type.shape) 
            print("tkg[1] shape",tkg[1].shape) 
            print("edge_attr shape",edge_attr.shape) 
            # raise ValueError("Values in the tensor are printted.") 
            gx = self.conv1(weight, edge, node_type, tkg[1], edge_attr).relu()
            #gx = self.bn1(gx)
            gx = self.conv2(gx, edge, node_type, tkg[1], edge_attr).relu()
            #gx = self.bn2(gx)
            #gx = self.conv3(gx, edge, tkg[1], edge_attr).relu()
            node_emb = self.lin(gx)
            #g_emb = self.graph_model(self.ent_embeddings.weight, edge, batch)[tkg[4].long()]
            g_emb = node_emb[tkg[4].long()]   
            self.tsne_emb = self.lin2(g_emb)    
            self.tsne_emb_all = self.lin2(node_emb)   
            g_emb = g_emb.unsqueeze(dim=0)
            contains_nan = torch.isnan(g_emb).any().item()
            if contains_nan:
                print("The tensor contains NaN values.",contains_nan)
                raise ValueError("Values in the tensor are Nan.") 
            x = torch.cat((x, g_emb), dim=2)

            pos = torch.sum(self.lin2(node_emb[tkg[0].long()]) + rel_emb + temp_emb - self.lin2(node_emb[tkg[2].long()]), dim=1)
            kg_loss = torch.mean(pos)
            
            self.ent_embeddings.weight.data.renorm_(p=2, dim=1, maxnorm=1).detach()
            self.rel_type_embeddings.weight.data.renorm_(p=2, dim=1, maxnorm=1).detach()
            self.month_embeddings.weight.data.renorm_(p=2, dim=1, maxnorm=1).detach()
            self.day_embeddings.weight.data.renorm_(p=2, dim=1, maxnorm=1).detach()
            self.hour_embeddings.weight.data.renorm_(p=2, dim=1, maxnorm=1).detach()
            self.minutes_embeddings.weight.data.renorm_(p=2, dim=1, maxnorm=1).detach()
            self.sec_embeddings.weight.data.renorm_(p=2, dim=1, maxnorm=1).detach()
            
        #
        if self.use_kg:
            print("we are at 7")
            kg_loss = self.kge(tkg[0], tkg[2], tkg[1], tkg[3])
            kg_emb = self.kge.ent_embeddings.weight[tkg[4].long()]
            self.kge.regularize_embeddings()
            kg_emb = kg_emb.unsqueeze(dim=0)
            x = torch.cat((x, kg_emb), dim=2)
        x = x.view(-1)
        #print(x.shape)
        price_pred = self.pred(x)
        # contains_nan = torch.isnan(price_pred).any().item()
        # if contains_nan:
        #     print("The tensor contains NaN values.",contains_nan)
        #     raise ValueError("Values in the tensor are Nan.")
        return price_pred, kg_loss


  